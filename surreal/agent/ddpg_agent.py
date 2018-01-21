"""
Actor function
"""
import torch
from torch.autograd import Variable
from .base import Agent, AgentMode
from surreal.model.ddpg_net import DDPGModel
import numpy as np
from .action_noise import *
from surreal.session import ConfigError

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
        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_dim = self.env_config.obs_spec.dim[0]
        self.use_z_filter = self.learner_config.algo.use_z_filter
        

        self.model = DDPGModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            use_z_filter=self.use_z_filter,
        )
        
        self.init_noise()


    def init_noise(self):
        if self.agent_mode is AgentMode.eval_deterministic:
            return
        self.noise_type = self.learner_config.algo.exploration.noise_type
        if self.noise_type == 'normal':
            self.noise = NormalActionNoise(np.zeros(self.action_dim), np.ones(self.action_dim) * self.learner_config.algo.exploration.sigma)
        elif self.noise_type == 'ou_noise':
            self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim), 
                                                    sigma=self.learner_config.algo.exploration.sigma,
                                                    theta=self.learner_config.algo.exploration.theta,
                                                    dt=self.learner_config.algo.exploration.dt)
        else:
            raise ConfigError('Noise type {} undefined.'.format(self.noise_type))

    def act(self, obs):

        assert torch.is_tensor(obs)
        obs = Variable(obs.unsqueeze(0))
        action = self.model.actor(obs).data.numpy().squeeze()

        if self.agent_mode is not AgentMode.eval_deterministic:
            action += self.noise()

        np.clip(action, -1, 1, out=action)
        return action

    def module_dict(self):
        return {
            'ddpg': self.model,
        }

    def default_config(self):
        return {
            'model': {
                'convs': '_list_',
                'fc_hidden_sizes': '_list_',
            },
        }

    def reset(self):
        if self.agent_mode is not AgentMode.eval_deterministic:
            self.noise.reset()