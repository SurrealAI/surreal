import torch
import torch.nn as nn
import numpy as np
from .base import Learner
from .aggregator import SSARAggregator, FrameStackPreprocessor
from surreal.model.ddpg_net import DDPGModel
from surreal.session import BASE_LEARNER_CONFIG, ConfigError
import surreal.utils as U
import torchx as tx


class DDPGLearner(Learner):
    '''
    DDPGLearner: subclass of Learner that contains DDPG algorithm logic
    Attributes:
        gpu_option: 'cpu' if not using GPU, 'cuda:all' otherwise
        model: instance of DDPGModel from surreal.model.ddpg_net
        model_target: instance of DDPGModel, used as a reference policy
            for Bellman updates
        use_action_regularization: boolean flag -- regularization method based on
            https://arxiv.org/pdf/1802.09477.pdf
        use_double_critic: boolean flag -- overestimation bias correction based on
            https://arxiv.org/pdf/1802.09477.pdf
        [actor/critic]_optim: Adam Optimizer for policy and baseline network
        aggregator: experience aggregator used to batch experiences from
            a list of experiences into a format usable by the model.
            For available aggregators, see surreal.learner.aggregator
        target_update_type: 'hard' update sets the weights of model_target equal to model
            after target_update_interval steps, whereas 'soft' update moves the parameters
            of model_target towards model after every step
        total_learn_time, forward_time, etc: timers that measure average time spent in
            each operation of the learner. These timers will be reported in tensorboard.

    important member functions:
        private methods:
        _optimize: function that makes policy and value function updates

        public methods:
        learn: method to perform optimization and send to tensorplex for log
        module_dict: returns the corresponding parameters
        preprocess: this function is called in learner/main prior to learn(),
            This operation occurs in a separate thread, meaning that conversion
            from numpy arrays to gpu tensors can occur asynchronously to gpu
            processing operations in learn().

    Arguments:
        learner_config, env_config, session_config: experiment setup configurations.  An example set of configs
            can be found at surreal/main/ddpg_configs.py.  Note that the surreal/env/make_env function adds attributes
            env_config.action_spec and env_config.obs_spec, which are required for this init method to function
            properly.
    '''

    def __init__(self, learner_config, env_config, session_config):
        super().__init__(learner_config, env_config, session_config)

        self.current_iteration = 0

        # load multiple optimization instances onto a single gpu
        self.batch_size = self.learner_config.replay.batch_size
        self.discount_factor = self.learner_config.algo.gamma
        self.n_step = self.learner_config.algo.n_step
        self.is_pixel_input = self.env_config.pixel_input
        self.use_layernorm = self.learner_config.model.use_layernorm
        self.use_double_critic = self.learner_config.algo.network.use_double_critic
        self.use_action_regularization = self.learner_config.algo.network.use_action_regularization

        self.frame_stack_concatenate_on_env = self.env_config.frame_stack_concatenate_on_env

        self.log.info('Initializing DDPG learner')
        self._num_gpus = session_config.learner.num_gpus
        if not torch.cuda.is_available():
            self.gpu_ids = 'cpu'
            self.log.info('Using CPU')
        else:
            self.gpu_ids = 'cuda:all'
            self.log.info('Using GPU')
            self.log.info('cudnn version: {}'.format(torch.backends.cudnn.version()))
            torch.backends.cudnn.benchmark = True
            self._num_gpus = 1

        with tx.device_scope(self.gpu_ids):
            self._target_update_init()

            self.clip_actor_gradient = self.learner_config.algo.network.clip_actor_gradient
            if self.clip_actor_gradient:
                self.actor_gradient_clip_value = self.learner_config.algo.network.actor_gradient_value_clip
                self.log.info('Clipping actor gradient at {}'.format(self.actor_gradient_clip_value))

            self.clip_critic_gradient = self.learner_config.algo.network.clip_critic_gradient
            if self.clip_critic_gradient:
                self.critic_gradient_clip_value = self.learner_config.algo.network.critic_gradient_value_clip
                self.log.info('Clipping critic gradient at {}'.format(self.critic_gradient_clip_value))

            self.action_dim = self.env_config.action_spec.dim[0]
            self.model = DDPGModel(
                obs_spec=self.env_config.obs_spec,
                action_dim=self.action_dim,
                use_layernorm=self.use_layernorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
                conv_out_channels=self.learner_config.model.conv_spec.out_channels,
                conv_kernel_sizes=self.learner_config.model.conv_spec.kernel_sizes,
                conv_strides=self.learner_config.model.conv_spec.strides,
                conv_hidden_dim=self.learner_config.model.conv_spec.hidden_output_dim,
            )

            self.model_target = DDPGModel(
                obs_spec=self.env_config.obs_spec,
                action_dim=self.action_dim,
                use_layernorm=self.use_layernorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
                conv_out_channels=self.learner_config.model.conv_spec.out_channels,
                conv_kernel_sizes=self.learner_config.model.conv_spec.kernel_sizes,
                conv_strides=self.learner_config.model.conv_spec.strides,
                conv_hidden_dim=self.learner_config.model.conv_spec.hidden_output_dim,
            )

            if self.use_double_critic:
                self.model2 = DDPGModel(
                    obs_spec=self.env_config.obs_spec,
                    action_dim=self.action_dim,
                    use_layernorm=self.use_layernorm,
                    actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                    critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
                    conv_out_channels=self.learner_config.model.conv_spec.out_channels,
                    conv_kernel_sizes=self.learner_config.model.conv_spec.kernel_sizes,
                    conv_strides=self.learner_config.model.conv_spec.strides,
                    conv_hidden_dim=self.learner_config.model.conv_spec.hidden_output_dim,
                    critic_only=True,
                )

                self.model_target2 = DDPGModel(
                    obs_spec=self.env_config.obs_spec,
                    action_dim=self.action_dim,
                    use_layernorm=self.use_layernorm,
                    actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                    critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
                    conv_out_channels=self.learner_config.model.conv_spec.out_channels,
                    conv_kernel_sizes=self.learner_config.model.conv_spec.kernel_sizes,
                    conv_strides=self.learner_config.model.conv_spec.strides,
                    conv_hidden_dim=self.learner_config.model.conv_spec.hidden_output_dim,
                    critic_only=True,
                )

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

            if self.use_double_critic:
                self.log.info('Using Adam for critic with learning rate {}'.format(self.learner_config.algo.network.lr_critic))
                self.critic_optim2 = torch.optim.Adam(
                    self.model2.get_critic_parameters(),
                    lr=self.learner_config.algo.network.lr_critic,
                    weight_decay=self.learner_config.algo.network.critic_regularization # Weight regularization term
                )

            self.log.info('Using {}-step bootstrapped return'.format(self.learner_config.algo.n_step))
            self.frame_stack_preprocess = FrameStackPreprocessor(self.env_config.frame_stacks)
            self.aggregator = SSARAggregator(self.env_config.obs_spec, self.env_config.action_spec)

            self.model_target.actor.hard_update(self.model.actor)
            self.model_target.critic.hard_update(self.model.critic)

            if self.use_double_critic:
                self.model_target2.critic.hard_update(self.model2.critic)

            self.total_learn_time = U.TimeRecorder()
            self.forward_time = U.TimeRecorder()
            self.critic_update_time = U.TimeRecorder()
            self.actor_update_time = U.TimeRecorder()

    # override
    def preprocess(self, batch):
        '''
        Override for learner/base/preprocess.  Before learn() is called, preprocess() takes the batch and converts
        the numpy arrays to pytorch tensors.  Note that this operation will transfer the data to gpu if a gpu is used.

        Arguments:
            batch: a batch of numpy arrays from the replay memory
        '''
        # Convert all numpy arrays to pytorch tensors, and transfers to gpu if applicable
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

    def _optimize(self, obs, actions, rewards, obs_next, done):
        '''
        Note that while the replay contains uint8, the
        aggregator returns float32 tensors

        Arguments:
            obs: an observation from the minibatch, often represented as s_n in literature. Dimensionality: (N, C) for
                low dimensional inputs, (N, C, H, W) for pixel inputs
            actions: actions taken given observations obs, often represented as a_n in literature.
                Dimensionality: (N, A), where A is the dimensionality of a single action
            rewards: rewards received after action is taken. Dimensionality: N
            obs_next: an observation from the minibatch, often represented as s_{n+1} in literature
            done: 1 if obs_next is terminal, 0 otherwise. Dimensionality: N
        '''
        with tx.device_scope(self.gpu_ids):

            with self.forward_time.time():
                assert actions.max().item() <= 1.0
                assert actions.min().item() >= -1.0

                # estimate rewards using the next state: r + argmax_a Q'(s_{t+1}, u'(a))

                model_policy, next_Q_target = self.model_target.forward(obs_next)
                if self.use_action_regularization:
                    # https://github.com/sfujim/TD3/blob/master/TD3.py -- action regularization
                    policy_noise = 0.2
                    noise_clip = 0.5
                    batch_size = self.batch_size
                    noise = np.clip(np.random.normal(0, policy_noise, size=(batch_size, self.action_dim)), -noise_clip,
                                    noise_clip)
                    device_name = 'cpu'
                    if self._num_gpus > 0:
                        device_name = 'cuda'
                    model_policy += torch.tensor(noise, dtype=torch.float32).to(device_name).detach()
                    model_policy = model_policy.clamp(-1, 1).to(device_name)
                y = rewards + pow(self.discount_factor, self.n_step) * next_Q_target * (1.0 - done)
                if self.use_double_critic:
                    _, next_Q_target2 = self.model_target2.forward(obs_next, action=model_policy)
                    y2 = rewards + pow(self.discount_factor, self.n_step) * next_Q_target2 * (1.0 - done)
                    y = torch.min(y, y2)
                y = y.detach()

                # compute Q(s_t, a_t)
                perception = self.model.forward_perception(obs)
                y_policy = self.model.forward_critic(
                    perception,
                    actions.detach()
                )

                y_policy2 = None
                if self.use_double_critic:
                    perception2 = self.model2.forward_perception(obs)
                    y_policy2 = self.model2.forward_critic(
                        perception2,
                        actions.detach()
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

                if self.use_double_critic:
                    self.model2.critic.zero_grad()
                    if self.is_pixel_input:
                        self.model2.perception.zero_grad()
                    critic_loss = self.critic_criterion(y_policy2, y)
                    critic_loss.backward()
                    if self.clip_critic_gradient:
                        self.model2.critic.clip_grad_value(self.critic_gradient_clip_value)
                    self.critic_optim2.step()

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
            if self.use_double_critic:
                tensorplex_update_dict['Q_policy2'] = y_policy2.mean().item()

            # (possibly) update target networks
            self._target_update()

            return tensorplex_update_dict

    def learn(self, batch):
        '''
        Performs a gradient descent step on 'batch'

        Arguments:
            batch: a minibatch sampled from the replay memory, after preprocessing steps such as transfer to pytorch
            tensors and aggregation step
        '''
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

    def _target_update_init(self):
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

    def _target_update(self):
        '''
        Perform update on target model networks.  This update is either 'soft', meaning the target model drifts towards
        the current model at a rate tau, or 'hard', meaning the target model performs a hard copy operation on the
        current model every target_update_interval steps.
        '''
        if self.target_update_type == 'soft':
            self.model_target.actor.soft_update(self.model.actor, self.target_update_tau)
            self.model_target.critic.soft_update(self.model.critic, self.target_update_tau)
            if self.use_double_critic:
                self.model_target2.critic.soft_update(self.model2.critic, self.target_update_tau)
                if self.is_pixel_input:
                    self.model_target2.perception.soft_update(self.model2.perception, self.target_update_tau)
            if self.is_pixel_input:
                self.model_target.perception.soft_update(self.model.perception, self.target_update_tau)
        elif self.target_update_type == 'hard':
            self.target_update_counter += 1
            if self.target_update_counter % self.target_update_interval == 0:
                self.model_target.actor.hard_update(self.model.actor)
                self.model_target.critic.hard_update(self.model.critic)
                if self.use_double_critic:
                    self.model_target2.critic.hard_update(self.model2.critic)
                    if self.is_pixel_input:
                        self.model_target2.perception.hard_update(self.model2.perception)
                if self.is_pixel_input:
                    self.model_target.perception.hard_update(self.model.perception)

    # override
    def _prefetcher_preprocess(self, batch):
        '''
        If frame_stack_preprocess is not set, each experience in the replay will be stored as a list of frames, as
        opposed to a single numpy array.  We must condense them into a single numpy array as that is what the
        aggregator expects.
        '''
        if not self.frame_stack_concatenate_on_env:
            batch = self.frame_stack_preprocess.preprocess_list(batch)
        batch = self.aggregator.aggregate(batch)
        return batch
