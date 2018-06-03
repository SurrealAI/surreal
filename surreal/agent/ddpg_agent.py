"""
Actor function
"""
import copy
import torch
import collections
from .base import Agent
from surreal.distributed import ModuleDict
from surreal.model.ddpg_net import DDPGModel
import numpy as np
from .action_noise import *
from surreal.session import ConfigError
import time
import torchx as tx
import torchx.nn as nnx
from .param_noise import NormalParameterNoise, AdaptiveNormalParameterNoise

class DDPGAgent(Agent):

    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 agent_id,
                 agent_mode):

        super().__init__(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config,
            agent_id=agent_id,
            agent_mode=agent_mode,
        )

        self.agent_id = agent_id
        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_spec = self.env_config.obs_spec
        self.use_z_filter = self.learner_config.algo.use_z_filter
        self.use_layernorm = self.learner_config.model.use_layernorm
        self.sleep_time = self.env_config.agent_sleep_time

        self.param_noise = None
        self.param_noise_type = self.learner_config.algo.exploration.param_noise_type
        self.param_noise_sigma = self.learner_config.algo.exploration.param_noise_sigma
        self.param_noise_alpha = self.learner_config.algo.exploration.param_noise_alpha
        self.param_noise_target_stddev = self.learner_config.algo.exploration.param_noise_target_stddev

        self.noise_type = self.learner_config.algo.exploration.noise_type
        if env_config.num_agents == 1:
            # If only one agent, we don't want a sigma of 0
            self.sigma = self.learner_config.algo.exploration.max_sigma / 3.0
        else:
            self.sigma = self.learner_config.algo.exploration.max_sigma * (float(agent_id) / (env_config.num_agents))
        print('Using exploration sigma', self.sigma)

        self._num_gpus = session_config.agent.num_gpus
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
            self.model = DDPGModel(
                obs_spec=self.obs_spec,
                action_dim=self.action_dim,
                use_layernorm=self.use_layernorm,
                actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
                critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
                use_z_filter=self.use_z_filter,
            )
            self.model.eval()

            self.init_noise()

    def init_noise(self):
        """
            initializes exploration noise
            and populates self.noise, a callable that returns noise of dimension same as action
        """
        if self.agent_mode == 'eval_deterministic':
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
            obs_variable = collections.OrderedDict()
            for modality in obs:
                modality_dict = collections.OrderedDict()
                for key in obs[modality]:
                    modality_dict[key] = torch.tensor(obs[modality][key], dtype=torch.float32).unsqueeze(0)
                obs_variable[modality] = modality_dict
            action, _ = self.model(obs_variable, calculate_value=False)
            if self.param_noise and self.param_noise_type == 'adaptive_normal':
                self.param_noise.compute_action_distance(obs_variable, action)
            action = action.data.cpu().numpy()[0]

            action = action.clip(-1, 1)

            if self.agent_mode != 'eval_deterministic':
                action += self.noise()

            action = action.clip(-1, 1)
            return action

    def module_dict(self, model=None):
        # My default, module_dict refers to the module_dict for the current model.  But, you can
        # generate a module_dict for other models as well -- e.g. param_noise uses a separate module_dict
        # to calculate action difference
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
        if self.agent_mode != 'eval_deterministic':
            self.noise.reset()

