"""
Actor function
"""
import torch
import collections
from .base import Agent
from surreal.model.ddpg_net import DDPGModel
import numpy as np
from .action_noise import *
from surreal.session import ConfigError
import time
import torchx as tx
import torchx.nn as nnx

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
        
        self.noise_type = self.learner_config.algo.exploration.noise_type
        if type(self.learner_config.algo.exploration.sigma) == list:
            # Use mod to wrap around the list of sigmas if the number of agents is greater than the length of the array
            self.sigma = self.learner_config.algo.exploration.sigma[agent_id % len(self.learner_config.algo.exploration.sigma)]
        elif type(self.learner_config.algo.exploration.sigma) in [int, float]:
            self.sigma = self.learner_config.algo.exploration.sigma
        else:
            raise ConfigError('Sigma {} undefined.'.format(self.learner_config.algo.exploration.sigma))

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
            action, _ = self.model.forward(obs_variable, calculate_value=False)
            action = action.data.cpu().numpy()[0]
            perception = self.model.forward_perception(obs_variable)
            action = self.model.forward_actor(perception).data.cpu().numpy()[0]

            action = action.clip(-1, 1)

            if self.agent_mode != 'eval_deterministic':
                action += self.noise()

            action = action.clip(-1, 1)
            return action

    def module_dict(self):
        return {
            'ddpg': self.model,
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

