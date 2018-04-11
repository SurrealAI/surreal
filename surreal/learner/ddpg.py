from queue import Queue
import torch
import torch.nn as nn
import numpy as np
import itertools
from .base import Learner
from .aggregator import SSARAggregator, NstepReturnAggregator, MultistepAggregator
from surreal.model.ddpg_net import DDPGModel
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG
from surreal.session import BASE_LEARNER_CONFIG, ConfigError
from surreal.utils.pytorch import GpuVariable as Variable
import surreal.utils as U


class DDPGLearner(Learner):

    def __init__(self, learner_config, env_config, session_config):
        super().__init__(learner_config, env_config, session_config)

        self.current_iteration = 0

        # load multiple optimization instances onto a single gpu
        self.batch_queue_size = 5
        self.batch_queue = Queue(maxsize=self.batch_queue_size)

        self.discount_factor = self.learner_config.algo.gamma
        self.n_step = self.learner_config.algo.n_step
        self.is_uint8_pixel_input = self.learner_config.algo.is_uint8_pixel_input
        self.use_z_filter = self.learner_config.algo.use_z_filter
        self.use_batchnorm = self.learner_config.algo.use_batchnorm
        self.use_layernorm = self.learner_config.algo.use_layernorm

        self.log.info('Initializing DDPG learner')
        num_gpus = session_config.learner.num_gpus
        self.gpu_ids = list(range(num_gpus))

        if not self.gpu_ids:
            self.log.info('Using CPU')
        else:
            self.log.info('Using GPU: {}'.format(self.gpu_ids))
            self.log.info('cudnn version: {}'.format(torch.backends.cudnn.version()))
            torch.backends.cudnn.benchmark = True

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
            self.obs_dim = self.env_config.obs_spec.dim

            self.model = DDPGModel(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                is_uint8_pixel_input=self.is_uint8_pixel_input,
                use_z_filter=self.use_z_filter,
                use_batchnorm=self.use_batchnorm,
                use_layernorm=self.use_layernorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
            )
            # self.model.train()

            self.model_target = DDPGModel(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                is_uint8_pixel_input=self.is_uint8_pixel_input,
                use_z_filter=self.use_z_filter,
                use_batchnorm=self.use_batchnorm,
                use_layernorm=self.use_layernorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
            )
            # self.model.eval()

            self.critic_criterion = nn.MSELoss()

            self.log.info('Using Adam for critic with learning rate {}'.format(self.learner_config.algo.lr_critic))
            self.critic_optim = torch.optim.Adam(
                itertools.chain(self.model.critic.parameters(), self.model.perception.parameters()),
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
            
            self.aggregate_and_preprocess_time = U.TimeRecorder()
            self.total_learn_time = U.TimeRecorder()
            self.data_prep_time = U.TimeRecorder()
            self.forward_time = U.TimeRecorder()

            self.critic_update_time = U.TimeRecorder()
            self.critic_optim_time = U.TimeRecorder()
            self.critic_forward_time = U.TimeRecorder()
            self.critic_backward_time = U.TimeRecorder()
            self.critic_gradient_clip_time = U.TimeRecorder()

            self.actor_update_time = U.TimeRecorder()
            self.actor_optim_time = U.TimeRecorder()
            self.actor_forward_time = U.TimeRecorder()
            self.actor_backward_time = U.TimeRecorder()
            self.actor_gradient_clip_time = U.TimeRecorder()

    def preprocess(self, obs, actions, rewards, obs_next, done):
        with U.torch_gpu_scope(self.gpu_ids):
            visual_obs, flat_obs = obs

            if visual_obs is not None:
                assert torch.is_tensor(visual_obs)
                visual_obs = Variable(visual_obs).detach()

            if flat_obs is not None:
                assert torch.is_tensor(flat_obs)
                flat_obs = Variable(flat_obs).detach()

            obs = (visual_obs, flat_obs)
            visual_obs_next, flat_obs_next = obs_next

            if visual_obs_next is not None:
                assert torch.is_tensor(visual_obs_next)
                visual_obs_next = Variable(visual_obs_next).detach()

            if flat_obs_next is not None:
                assert torch.is_tensor(flat_obs_next)
                flat_obs_next = Variable(flat_obs_next).detach()

            obs_next = (visual_obs_next, flat_obs_next)
            #print('preprocess')
            #print(type(obs_next[0].data))

            actions = Variable(actions)
            rewards = Variable(rewards)
            done = Variable(done)
            return (obs, actions, rewards, obs_next, done)

    def _optimize(self, obs, actions, rewards, obs_next, done):
        '''
        obs is a tuple (visual_obs, flat_obs). If visual_obs is not None, it is a FloatTensor
        of observations, (N, C, H, W).  Note that while the replay contains uint8, the
        aggregator returns float32 tensors
        '''
        with U.torch_gpu_scope(self.gpu_ids):

            with self.forward_time.time():

                assert actions.max().data[0] <= 1.0
                assert actions.min().data[0] >= -1.0

                # estimate rewards using the next state: r + argmax_a Q'(s_{t+1}, u'(a))
                # obs_next.volatile = True
                perception_next_target = self.model_target.forward_perception(obs_next)
                next_actions_target = self.model_target.forward_actor(perception_next_target)

                # obs_next.volatile = False
                next_Q_target = self.model_target.forward_critic(perception_next_target,
                                                                 next_actions_target)
                # next_Q_target.volatile = False
                y = rewards + pow(self.discount_factor, self.n_step) * next_Q_target * (1.0 - done)
                y = y.detach()

                # compute Q(s_t, a_t)
                perception = self.model.forward_perception(obs)
                perception_next = self.model.forward_perception(obs_next)
                y_policy = self.model.forward_critic(
                    perception,
                    actions.detach() # TODO: why do we detach here
                )
                torch.cuda.synchronize()

            profile_critic = True
            profile_actor = True
            profile_gpu = True

            # critic update
            with self.critic_update_time.time():
                self.model.critic.zero_grad()
                self.model.perception.zero_grad()
                critic_loss = self.critic_criterion(y_policy, y)        
                critic_loss.backward()
                if self.clip_critic_gradient:
                    self.model.critic.clip_grad_value(self.critic_gradient_clip_value)
                self.critic_optim.step()
                torch.cuda.synchronize()

            # actor update
            with self.actor_update_time.time():
                self.model.actor.zero_grad()
                actor_loss = -self.model.forward_critic(
                    perception.detach(),
                    self.model.forward_actor(perception.detach())
                )
                actor_loss = actor_loss.mean()
                actor_loss.backward()
                if self.clip_actor_gradient:
                    self.model.actor.clip_grad_value(self.actor_gradient_clip_value)
                self.actor_optim.step()
                torch.cuda.synchronize()

            tensorplex_update_dict = {
                'actor_loss': actor_loss.data[0],
                'critic_loss': critic_loss.data[0],
                'action_norm': actions.norm(2, 1).mean().data[0],
                'rewards': rewards.mean().data[0],
                'Q_target': y.mean().data[0],
                'Q_policy': y_policy.mean().data[0],
                'performance/data_prep_time': self.data_prep_time.avg,
                'performance/forward_time': self.forward_time.avg,
                'performance/critic_update_time': self.critic_update_time.avg,
                # # 'performance/critic_optim_time': self.critic_optim_time.avg,
                # 'performance/critic_forward_time': self.critic_forward_time.avg,
                # 'performance/critic_backward_time': self.critic_backward_time.avg,
                # # 'performance/critic_gradient_clip_time': self.critic_gradient_clip_time.avg,
                'performance/actor_update_time': self.actor_update_time.avg,
                # # 'performance/actor_optim_time': self.actor_optim_time.avg,
                # 'performance/actor_forward_time': self.actor_forward_time.avg,
                # 'performance/actor_backward_time': self.actor_backward_time.avg,
                # # 'performance/actor_gradient_clip_time': self.actor_gradient_clip_time.avg,
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
        with self.aggregate_and_preprocess_time.time():
            batch = self.aggregator.aggregate(batch)
            # The preprocess step creates Variables which will become GpuVariables
            batch.obs, batch.actions, batch.rewards, batch.obs_next, batch.dones = self.preprocess(
                batch.obs,
                batch.actions,
                batch.rewards,
                batch.obs_next, 
                batch.dones
            )
            self.batch_queue.put(batch)
        if self.batch_queue.qsize() == self.batch_queue_size:
            with self.total_learn_time.time():
                batch = self.batch_queue.get()

                tensorplex_update_dict = self._optimize(
                    batch.obs,
                    batch.actions,
                    batch.rewards,
                    batch.obs_next,
                    batch.dones
                )
                tensorplex_update_dict['performance/total_learn_time'] = self.total_learn_time.avg
                tensorplex_update_dict['performance/aggregate_and_preprocess_time'] = self.aggregate_and_preprocess_time.avg
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
            U.soft_update(self.model_target.perception, self.model.perception, self.target_update_tau)
        elif self.target_update_type == 'hard':
            self.target_update_counter += 1
            if self.target_update_counter % self.target_update_interval == 0:
                U.hard_update(self.model_target.actor, self.model.actor)
                U.hard_update(self.model_target.critic, self.model.critic)
                U.hard_update(self.model_target.perception, self.model.perception)
