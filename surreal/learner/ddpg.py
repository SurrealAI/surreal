from queue import Queue
import torch
import torch.nn as nn
import numpy as np
import itertools
from .base import Learner
from .aggregator import NstepReturnAggregator, SSARAggregator, FrameStackPreprocessor
from surreal.model.ddpg_net import DDPGModel
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG
from surreal.session import BASE_LEARNER_CONFIG, ConfigError
#from surreal.utils.pytorch import #GpuVariable as Variable
import surreal.utils as U
import torchx as tx
import torchx.nn as nnx


class DDPGLearner(Learner):

    def __init__(self, learner_config, env_config, session_config):
        super().__init__(learner_config, env_config, session_config)

        self.current_iteration = 0

        # load multiple optimization instances onto a single gpu
        self.batch_queue_size = 5
        self.batch_queue = Queue(maxsize=self.batch_queue_size)

        self.discount_factor = self.learner_config.algo.gamma
        self.n_step = self.learner_config.algo.n_step
        self.is_pixel_input = self.env_config.pixel_input
        self.use_z_filter = self.learner_config.algo.use_z_filter
        self.use_layernorm = self.learner_config.model.use_layernorm

        self.frame_stack_concatenate_on_agent = self.env_config.frame_stack_concatenate_on_agent

        self.log.info('Initializing DDPG learner')
        self._num_gpus = session_config.learner.num_gpus
        if self._num_gpus == 0:
            self.gpu_ids = 'cpu'
        else:
            self.gpu_ids = 'cuda:all'

        if self._num_gpus == 0:
            self.log.info('Using CPU')
        else:
            self.log.info('Using {} GPUs'.format(self._num_gpus))
            self.log.info('cudnn version: {}'.format(torch.backends.cudnn.version()))
            torch.backends.cudnn.benchmark = True

        with tx.device_scope(self.gpu_ids):
            self.target_update_init()

            self.clip_actor_gradient = self.learner_config.algo.network.clip_actor_gradient
            if self.clip_actor_gradient:
                self.actor_gradient_clip_value = self.learner_config.algo.network.actor_gradient_norm_clip
                self.log.info('Clipping actor gradient at {}'.format(self.actor_gradient_clip_value))

            self.clip_critic_gradient = self.learner_config.algo.network.clip_critic_gradient
            if self.clip_critic_gradient:
                self.critic_gradient_clip_value = self.learner_config.algo.network.critic_gradient_norm_clip
                self.log.info('Clipping critic gradient at {}'.format(self.critic_gradient_clip_value))

            self.action_dim = self.env_config.action_spec.dim[0]
            self.model = DDPGModel(
                obs_spec=self.env_config.obs_spec,
                action_dim=self.action_dim,
                use_layernorm=self.use_layernorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
                use_z_filter=self.use_z_filter,
            )
            # self.model.train()

            self.model_target = DDPGModel(
                obs_spec=self.env_config.obs_spec,
                action_dim=self.action_dim,
                use_layernorm=self.use_layernorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
                use_z_filter=self.use_z_filter,
            )
            # self.model.eval()

            self.critic_criterion = nn.MSELoss()

            self.log.info('Using Adam for critic with learning rate {}'.format(self.learner_config.algo.network.lr_critic))
            self.critic_optim = torch.optim.Adam(
                self.model.get_critic_parameters(),
                lr=self.learner_config.algo.network.lr_critic,
                weight_decay=self.learner_config.algo.network.critic_regularization # Weight regularization term
            )

            self.log.info('Using Adam for actor with learning rate {}'.format(self.learner_config.algo.network.lr_actor))
            self.actor_optim = torch.optim.Adam(
                self.model.get_actor_parameters(),
                lr=self.learner_config.algo.network.lr_actor,
                weight_decay=self.learner_config.algo.network.actor_regularization # Weight regularization term
            )

            self.log.info('Using {}-step bootstrapped return'.format(self.learner_config.algo.n_step))
            # Note that the Nstep Return aggregator does not care what is n. It is the experience sender that cares
            self.frame_stack_preprocess = FrameStackPreprocessor(self.env_config.frame_stacks)
            self.aggregator = SSARAggregator(self.env_config.obs_spec, self.env_config.action_spec)

            self.model_target.actor.hard_update(self.model.actor)
            self.model_target.critic.hard_update(self.model.critic)
            # self.train_iteration = 0
            
            self.total_learn_time = U.TimeRecorder()
            self.forward_time = U.TimeRecorder()
            self.critic_update_time = U.TimeRecorder()
            self.actor_update_time = U.TimeRecorder()

    def preprocess(self, batch):
        with tx.device_scope(self.gpu_ids):
            obs, actions, rewards, obs_next, done = (
                batch['obs'],
                batch['actions'],
                batch['rewards'],
                batch['obs_next'],
                batch['dones']
            )
            device_name = 'cpu'
            if self._num_gpus > 0:
                device_name = 'cuda'

            for modality in obs:
                for key in obs[modality]:
                    if modality == 'pixel':
                        obs[modality][key] = (torch.tensor(obs[modality][key], dtype=torch.uint8)
                            .to(torch.device(device_name))).float().detach()
                    else:
                        obs[modality][key] = (torch.tensor(obs[modality][key], dtype=torch.float32)
                            .to(torch.device(device_name))).detach()

            for modality in obs_next:
                for key in obs_next[modality]:
                    if modality == 'pixel':
                        obs_next[modality][key] = (torch.tensor(obs_next[modality][key], dtype=torch.uint8)
                            .to(torch.device(device_name))).float().detach()
                    else:
                        obs_next[modality][key] = (torch.tensor(obs_next[modality][key], dtype=torch.float32)
                            .to(torch.device(device_name))).detach()

            actions = torch.tensor(actions, dtype=torch.float32).to(torch.device(device_name))
            rewards = torch.tensor(rewards, dtype=torch.float32).to(torch.device(device_name))
            done = torch.tensor(done, dtype=torch.float32).to(torch.device(device_name))

            (
                batch['obs'],
                batch['actions'],
                batch['rewards'],
                batch['obs_next'],
                batch['dones']
            ) = (
                obs,
                actions,
                rewards,
                obs_next,
                done
            )
            return batch

    def _assert_gpu(self, tensor, name):
        # Sometimes automatic conversion to cuda tensor in tx.device_scope
        # doesn't work, correct for that here
        if self._num_gpus == 0:
            return tensor
        if not tensor.is_cuda:
            print('----Expected cuda tensor, received cpu tensor------')
            print('name', name)
            print('is_cuda', tensor.is_cuda)
            print('tensor_type', type(tensor))
            print('dimensions', tensor.dim())
            print('iter', self.current_iteration)
            return tensor.to(torch.device('cuda'))
        return tensor

    def _optimize(self, obs, actions, rewards, obs_next, done):
        '''
        obs is a tuple (visual_obs, flat_obs). If visual_obs is not None, it is a FloatTensor
        of observations, (N, C, H, W).  Note that while the replay contains uint8, the
        aggregator returns float32 tensors
        '''
        with tx.device_scope(self.gpu_ids):

            with self.forward_time.time():
                for o in [obs, obs_next]:
                    for modality in o:
                        for k in o[modality]:
                            o[modality][k] = self._assert_gpu(o[modality][k], k)
                actions = self._assert_gpu(actions, 'actions')
                rewards = self._assert_gpu(rewards, 'rewards')
                done = self._assert_gpu(done, 'done')

                assert actions.max().item() <= 1.0
                assert actions.min().item() >= -1.0

                # estimate rewards using the next state: r + argmax_a Q'(s_{t+1}, u'(a))
                # obs_next.volatile = True
                _, next_Q_target = self.model_target.forward(obs_next)
                y = rewards + pow(self.discount_factor, self.n_step) * next_Q_target * (1.0 - done)
                y = y.detach()

                # compute Q(s_t, a_t)
                perception = self.model.forward_perception(obs)
                y_policy = self.model.forward_critic(
                    perception,
                    actions.detach() # TODO: why do we detach here
                )

            # critic update
            with self.critic_update_time.time():
                self.model.critic.zero_grad()
                if self.is_pixel_input:
                    self.model.perception.zero_grad()
                critic_loss = self.critic_criterion(y_policy, y)        
                critic_loss.backward()
                if self.clip_critic_gradient:
                    self.model.critic.clip_grad_value(self.critic_gradient_clip_value)
                self.critic_optim.step()

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

            tensorplex_update_dict = {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'action_norm': actions.norm(2, 1).mean().item(),
                'rewards': rewards.mean().item(),
                'Q_target': y.mean().item(),
                'Q_policy': y_policy.mean().item(),
                'performance/forward_time': self.forward_time.avg,
                'performance/critic_update_time': self.critic_update_time.avg,
                'performance/actor_update_time': self.actor_update_time.avg,
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
        with self.total_learn_time.time():
            tensorplex_update_dict = self._optimize(
                batch.obs,
                batch.actions,
                batch.rewards,
                batch.obs_next,
                batch.dones
            )
            tensorplex_update_dict['performance/total_learn_time'] = self.total_learn_time.avg
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
        target_update_config = self.learner_config.algo.network.target_update
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
            self.model_target.actor.soft_update(self.model.actor, self.target_update_tau)
            self.model_target.critic.soft_update(self.model.critic, self.target_update_tau)
            if self.is_pixel_input:
                self.model_target.perception.soft_update(self.model.perception, self.target_update_tau)
        elif self.target_update_type == 'hard':
            self.target_update_counter += 1
            if self.target_update_counter % self.target_update_interval == 0:
                self.model_target.actor.hard_update(self.model.actor)
                self.model_target.critic.hard_update(self.model.critic)
                if self.is_pixel_input:
                    self.model_target.perception.hard_update(self.model.perception)

    # override
    def _prefetch_thread_preprocess(self, batch):
        if not self.frame_stack_concatenate_on_agent:
            batch = self.frame_stack_preprocess.preprocess_list(batch)
        batch = self.aggregator.aggregate(batch)
        return batch

