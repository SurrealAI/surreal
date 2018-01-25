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

        self.target_update_init()

        self.discount_factor = self.learner_config.algo.gamma
        self.n_step = self.learner_config.algo.n_step
        self.use_z_filter = self.learner_config.algo.use_z_filter

        self.clip_actor_gradient = self.learner_config.algo.clip_actor_gradient
        if self.clip_actor_gradient:
            self.actor_gradient_clip_value = self.learner_config.algo.actor_gradient_clip_value

        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_dim = self.env_config.obs_spec.dim[0]

        self.model = DDPGModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            use_z_filter=self.use_z_filter,
        )

        self.model_target = DDPGModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            use_z_filter=self.use_z_filter,
        )

        self.critic_criterion = nn.MSELoss()

        self.critic_optim = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=1e-3
        )

        self.actor_optim = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=1e-4
        )

        # self.aggregator = NstepReturnAggregator(self.env_config.obs_spec, self.env_config.action_spec, self.discount_factor)
        self.aggregator = SSARAggregator(self.env_config.obs_spec, self.env_config.action_spec)


        U.hard_update(self.model_target.actor, self.model.actor)
        U.hard_update(self.model_target.critic, self.model.critic)
        # self.train_iteration = 0

    def _optimize(self, obs, actions, rewards, obs_next, done):
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

        self.update_tensorplex(tensorplex_update_dict)

        # (possibly) update target networks
        self.target_update()

    def learn(self, batch):
        batch = self.aggregator.aggregate(batch)
        self._optimize(
            batch.obs,
            batch.actions,
            batch.rewards,
            batch.obs_next,
            batch.dones
        )

    def module_dict(self):
        return {
            'ddpg': self.model,
        }

    def target_update_init(self):
        target_update_config = self.learner_config.algo.target_update
        self.target_update_type = target_update_config.type

        if self.target_update_type == 'soft':
            self.target_update_tau = target_update_config.tau
        elif self.target_update_type == 'hard':
            self.target_update_counter = 0
            self.target_update_interval = target_update_config.interval
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
