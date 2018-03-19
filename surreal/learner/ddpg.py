import torch
import torch.nn as nn
import numpy as np
from .base import Learner
from .aggregator import SSARAggregator, NstepReturnAggregator, MultistepAggregator
from surreal.model.ddpg_net import DDPGModel
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG, ConfigError
from surreal.utils.pytorch import GpuVariable as Variable
import surreal.utils as U


class DDPGLearner(Learner):

    def __init__(self, learner_config, env_config, session_config):
        super().__init__(learner_config, env_config, session_config)

        self.current_iteration = 0

        self.discount_factor = self.learner_config.algo.gamma
        self.n_step = self.learner_config.algo.n_step
        self.use_z_filter = self.learner_config.algo.use_z_filter
        self.use_batchnorm = self.learner_config.algo.use_batchnorm

        self.log.info('Initializing DDPG learner')
        num_gpus = session_config.learner.num_gpus
        self.gpu_ids = list(range(num_gpus))

        if not self.gpu_ids:
            self.log.info('Using CPU')
        else:
            self.log.info('Using GPU: {}'.format(self.gpu_ids))

        with U.torch_gpu_scope(self.gpu_ids):
            self.target_update_init()

            self.clip_actor_gradient = self.learner_config.algo.clip_actor_gradient
            if self.clip_actor_gradient:
                self.actor_gradient_clip_value = self.learner_config.algo.actor_gradient_clip_value
                self.log.info('Clipping actor gradient at {}'.format(self.actor_gradient_clip_value))

            self.clip_critic_gradient = self.learner_config.algo.clip_critic_gradient
            if self.clip_critic_gradient:
                self.critic_gradient_clip_value = self.learner_config.algo.critic_gradient_clip_value
                self.log.info('Clipping critic gradient at {}'.format(self.critic_gradient_clip_value))

            self.action_dim = self.env_config.action_spec.dim[0]
            self.obs_dim = self.env_config.obs_spec.dim[0]

            self.model = DDPGModel(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                use_z_filter=self.use_z_filter,
                use_batchnorm=self.use_batchnorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
            )
            self.model.train()

            self.model_target = DDPGModel(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                use_z_filter=self.use_z_filter,
                use_batchnorm=self.use_batchnorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
            )
            self.model.eval()

            self.critic_criterion = nn.MSELoss()

            self.log.info('Using Adam for critic with learning rate {}'.format(self.learner_config.algo.lr_critic))
            self.critic_optim = torch.optim.Adam(
                self.model.critic.parameters(),
                lr=self.learner_config.algo.lr_critic,
                weight_decay=self.learner_config.algo.critic_regularization # Weight regularization term
            )

            self.log.info('Using Adam for actor with learning rate {}'.format(self.learner_config.algo.lr_actor))
            self.actor_optim = torch.optim.Adam(
                self.model.actor.parameters(),
                lr=self.learner_config.algo.lr_actor,
                weight_decay=self.learner_config.algo.actor_regularization # Weight regularization term
            )

            self.log.info('Using {}-step bootstrapped return'.format(self.learner_config.algo.n_step))
            # Note that the Nstep Return aggregator does not care what is n. It is the experience sender that cares
            # self.aggregator = NstepReturnAggregator(self.env_config.obs_spec, self.env_config.action_spec, self.discount_factor)
            self.aggregator = SSARAggregator(self.env_config.obs_spec, self.env_config.action_spec)

            U.hard_update(self.model_target.actor, self.model.actor)
            U.hard_update(self.model_target.critic, self.model.critic)
            # self.train_iteration = 0

    def _optimize(self, obs, actions, rewards, obs_next, done):
        with U.torch_gpu_scope(self.gpu_ids):
            obs = Variable(obs)
            actions = Variable(actions)
            rewards = Variable(rewards)
            obs_next = Variable(obs_next)
            done = Variable(done)

            assert actions.max().data[0] <= 1.0
            assert actions.min().data[0] >= -1.0

            if self.use_z_filter:
                self.model.z_update(obs)

            # estimate rewards using the next state: r + argmax_a Q'(s_{t+1}, u'(a))
            # obs_next.volatile = True
            next_actions_target = self.model_target.forward_actor(obs_next)

            # obs_next.volatile = False
            next_Q_target = self.model_target.forward_critic(obs_next, next_actions_target)
            # next_Q_target.volatile = False
            y = rewards + pow(self.discount_factor, self.n_step) * next_Q_target * (1.0 - done)
            y = y.detach()

            # print('next_Q_target', next_Q_target)
            # print('y', y)

            # compute Q(s_t, a_t)
            y_policy = self.model.forward_critic(
                obs.detach(),
                actions.detach()
            )

            # critic update
            self.model.critic.zero_grad()
            critic_loss = self.critic_criterion(y_policy, y)
            critic_loss.backward()
            if self.clip_critic_gradient:
                self.model.critic.clip_grad_value(self.critic_gradient_clip_value)
            # for p in self.model.critic.parameters():
            #     p.grad.data.clamp_(-1.0, 1.0)
            self.critic_optim.step()

            # actor update
            self.model.actor.zero_grad()
            actor_loss = -self.model.forward_critic(
                obs.detach(),
                self.model.forward_actor(obs.detach())
            ).mean()
            actor_loss.backward()
            if self.clip_actor_gradient:
                self.model.actor.clip_grad_value(self.actor_gradient_clip_value)
            # for p in self.model.actor.parameters():
            #     p.grad.data.clamp_(-1.0, 1.0)
            self.actor_optim.step()

            tensorplex_update_dict = {
                'actor_loss': actor_loss.data[0],
                'critic_loss': critic_loss.data[0],
                'action_norm': actions.norm(2, 1).mean().data[0],
                'rewards': rewards.mean().data[0],
                'Q_target': y.mean().data[0],
                'Q_policy': y_policy.mean().data[0],
            }
            if self.use_z_filter:
                tensorplex_update_dict['observation_0_running_mean'] = self.model.z_filter.running_mean()[0]
                tensorplex_update_dict['observation_0_running_square'] =  self.model.z_filter.running_square()[0]
                tensorplex_update_dict['observation_0_running_std'] = self.model.z_filter.running_std()[0]            

            # (possibly) update target networks
            self.target_update()

            return tensorplex_update_dict

    def learn(self, batch):
        self.current_iteration += 1
        batch = self.aggregator.aggregate(batch)
        tensorplex_update_dict = self._optimize(
            batch.obs,
            batch.actions,
            batch.rewards,
            batch.obs_next,
            batch.dones
        )
        self.tensorplex.add_scalars(tensorplex_update_dict, global_step=self.current_iteration)
        self.periodic_checkpoint(
            global_steps=self.current_iteration,
            score=None,
        )

    def module_dict(self):
        return {
            'ddpg': self.model,
        }

    def checkpoint_attributes(self):
        return [
            'current_iteration',
            'model', 'model_target'
        ]

    def target_update_init(self):
        target_update_config = self.learner_config.algo.target_update
        self.target_update_type = target_update_config.type

        if self.target_update_type == 'soft':
            self.target_update_tau = target_update_config.tau
            self.log.info('Using soft target update with tau = {}'.format(self.target_update_tau))
        elif self.target_update_type == 'hard':
            self.target_update_counter = 0
            self.target_update_interval = target_update_config.interval
            self.log.info('Using hard target update every {} steps'.format(self.target_update_interval))
        else:
            raise ConfigError('Unsupported ddpg update type: {}'.format(target_update_config.type))

    def target_update(self):
        if self.target_update_type == 'soft':
            U.soft_update(self.model_target.actor, self.model.actor, self.target_update_tau)
            U.soft_update(self.model_target.critic, self.model.critic, self.target_update_tau)
        elif self.target_update_type == 'hard':
            self.target_update_counter += 1
            if self.target_update_counter % self.target_update_interval == 0:
                U.hard_update(self.model_target.actor, self.model.actor)
                U.hard_update(self.model_target.critic, self.model.critic)
