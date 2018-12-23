"""
DDPG actor class
"""
import copy
import collections
import time
import torch
import numpy as np
import torchx as tx
import torchx.nn as nnx
from surreal.distributed import ModuleDict
from surreal.model.ddpg_net import DDPGModel
from surreal.env import ExpSenderWrapperSSARNStepBootstrap
from surreal.session import ConfigError
from .base import Agent
from .action_noise import *
from .param_noise import NormalParameterNoise, AdaptiveNormalParameterNoise


class DDPGAgent(Agent):
    '''
    DDPGAgent: subclass of Agent that contains DDPG algorithm logic
    Attributes:
        model: A DDPG neural network model to generate actions from observations
        noise: A traditional action noise model. When called, noise produces an output
            of the same dimension as the action dimension
        param_noise: If not None, a parameter noise model, as outlined by
            https://blog.openai.com/better-exploration-with-parameter-noise/

    Important member functions:
        public methods:
        act: method to generate action from observation using the model
        module_dict: returns the corresponding parameters
        on_parameter_fetched: performs necessary updates to parameter noise
            given new parameters from the parameter server
        pre_episode: prepares model for new episode, performs update to the noise model
    '''

    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 agent_id,
                 agent_mode,
                 render=False):
        '''
        Constructor for DDPGAgent class.
        Important attributes:
            learner_config, env_config, session_config: experiment configurations
            agent_id: unique id in the range [0, num_agents)
            agent_mode: toggles between agent noise and deterministic behavior
        '''
        super().__init__(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config,
            agent_id=agent_id,
            agent_mode=agent_mode,
            render=render,
        )

        self.agent_id = agent_id
        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_spec = self.env_config.obs_spec
        self.use_layernorm = self.learner_config.model.use_layernorm
        self.sleep_time = self.env_config.sleep_time

        self.param_noise = None
        self.param_noise_type = self.learner_config.algo.exploration.param_noise_type
        self.param_noise_sigma = self.learner_config.algo.exploration.param_noise_sigma
        self.param_noise_alpha = self.learner_config.algo.exploration.param_noise_alpha
        self.param_noise_target_stddev = self.learner_config.algo.exploration.param_noise_target_stddev

        self.frame_stack_concatenate_on_env = self.env_config.frame_stack_concatenate_on_env

        self.noise_type = self.learner_config.algo.exploration.noise_type

        if env_config.num_agents == 1:
            # If only one agent, we don't want a sigma of 0
            self.sigma = self.learner_config.algo.exploration.max_sigma / 3.0
        else:
            self.sigma = self.learner_config.algo.exploration.max_sigma * (
            float(agent_id) / (env_config.num_agents))
        #self.sigma = self.learner_config.algo.exploration.sigma
        print('Using exploration sigma', self.sigma)

        if torch.cuda.is_available():
            self.gpu_ids = 'cuda:all'
            if self.agent_mode not in ['eval_deterministic_local', 'eval_stochastic_local']:
                self.log.info('DDPG agent is using GPU')
                # Note that user is responsible for only providing one GPU for the program
                self.log.info('cudnn version: {}'.format(torch.backends.cudnn.version()))
            torch.backends.cudnn.benchmark = True
        else:
            self.gpu_ids = 'cpu'
            if self.agent_mode not in ['eval_deterministic_local', 'eval_stochastic_local']:
                self.log.info('DDPG agent is using CPU')

        with tx.device_scope(self.gpu_ids):
            self.model = DDPGModel(
                obs_spec=self.obs_spec,
                action_dim=self.action_dim,
                use_layernorm=self.use_layernorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
                conv_out_channels=self.learner_config.model.conv_spec.out_channels,
                conv_kernel_sizes=self.learner_config.model.conv_spec.kernel_sizes,
                conv_strides=self.learner_config.model.conv_spec.strides,
                conv_hidden_dim=self.learner_config.model.conv_spec.hidden_output_dim,
            )
            self.model.eval()

            self._init_noise()

    def _init_noise(self):
        """
            initializes exploration noise and populates self.noise, a callable
            that returns noise of dimension same as action
        """
        if self.agent_mode in ['eval_deterministic', 'eval_deterministic_local']:
            return
        if self.noise_type == 'normal':
            self.noise = NormalActionNoise(
                np.zeros(self.action_dim),
                np.ones(self.action_dim) * self.sigma
            )
        elif self.noise_type == 'ou_noise':
            self.noise = OrnsteinUhlenbeckActionNoise(
                mu=np.zeros(self.action_dim),
                sigma=self.sigma,
                theta=self.learner_config.algo.exploration.theta,
                dt=self.learner_config.algo.exploration.dt
            )
        else:
            raise ConfigError('Noise type {} undefined.'.format(self.noise_type))
        if self.param_noise_type == 'normal':
            self.param_noise = NormalParameterNoise(self.param_noise_sigma)
        elif self.param_noise_type == 'adaptive_normal':
            model_copy = copy.deepcopy(self.model)
            module_dict_copy = ModuleDict(self.module_dict(model_copy))
            self.param_noise = AdaptiveNormalParameterNoise(
                model_copy,
                module_dict_copy,
                self.param_noise_target_stddev,
                alpha=self.param_noise_alpha,
                sigma=self.param_noise_sigma
            )

    def on_parameter_fetched(self, params, info):
        params = super().on_parameter_fetched(params, info)
        if self.param_noise:
            params = self.param_noise.apply(params)
        return params

    def act(self, obs):
        with tx.device_scope(self.gpu_ids):
            if self.sleep_time > 0.0:
                time.sleep(self.sleep_time)
            if not self.frame_stack_concatenate_on_env:
                # Output pixels of environment is a list of frames,
                # we concatenate the frames into a single numpy array
                obs = copy.deepcopy(obs)
                if 'pixel' in obs:
                    for key in obs['pixel']:
                        obs['pixel'][key] = np.concatenate(obs['pixel'][key], axis=0)
            # Convert to pytorch tensor
            obs_tensor = collections.OrderedDict()
            for modality in obs:
                modality_dict = collections.OrderedDict()
                for key in obs[modality]:
                    modality_dict[key] = torch.tensor(obs[modality][key], dtype=torch.float32).unsqueeze(0)
                obs_tensor[modality] = modality_dict
            action, _ = self.model(obs_tensor, calculate_value=False)
            if self.param_noise and self.param_noise_type == 'adaptive_normal':
                self.param_noise.compute_action_distance(obs_tensor, action)
            action = action.data.cpu().numpy()[0]

            action = action.clip(-1, 1)

            if self.agent_mode not in ['eval_deterministic', 'eval_deterministic_local']:
                action += self.noise()

            action = action.clip(-1, 1)
            return action

    def module_dict(self, model=None):
        # By default, module_dict refers to the module_dict for the current model.
        # But, you can generate a module_dict for other models as well --
        # e.g. param_noise uses a separate module_dict to calculate action difference
        if model == None:
            model = self.model
        return {
            'ddpg': model,
        }

    def default_config(self):
        return {
            'model': {
                'convs': '_list_',
                'actor_fc_hidden_sizes': '_list_',
                'critic_fc_hidden_sizes': '_list_',
            },
        }

    def pre_episode(self):
        super().pre_episode()
        if self.agent_mode not in ['eval_deterministic', 'eval_deterministic_local']:
            self.noise.reset()

    def prepare_env_agent(self, env):
        env = super().prepare_env_agent(env)
        env = ExpSenderWrapperSSARNStepBootstrap(env,
                                                 self.learner_config,
                                                 self.session_config)
        return env
