"""
Actor function
"""
import torch
from torch.autograd import Variable
from .base import Agent
import surreal.utils as U
from surreal.model.ddpg_net import DDPGModel
import numpy as np
from .action_noise import *
from surreal.session import ConfigError
import time

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
        self.obs_dim = self.env_config.obs_spec.dim
        self.use_z_filter = self.learner_config.algo.use_z_filter
        self.use_batchnorm = self.learner_config.algo.use_batchnorm
        self.sleep_time = self.learner_config.algo.agent_sleep_time
        
        self.noise_type = self.learner_config.algo.exploration.noise_type
        if type(self.learner_config.algo.exploration.sigma) == list:
            # Use mod to wrap around the list of sigmas if the number of agents is greater than the length of the array
            self.sigma = self.learner_config.algo.exploration.sigma[agent_id % len(self.learner_config.algo.exploration.sigma)]
        elif type(self.learner_config.algo.exploration.sigma) in [int, float]:
            self.sigma = self.learner_config.algo.exploration.sigma
        else:
            raise ConfigError('Sigma {} undefined.'.format(self.learner_config.algo.exploration.sigma))

        self.model = DDPGModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            use_z_filter=self.use_z_filter,
            use_batchnorm=self.use_batchnorm,
            actor_fc_hidden_sizes=self.learner_config.model.actor_fc_hidden_sizes,
            critic_fc_hidden_sizes=self.learner_config.model.critic_fc_hidden_sizes,
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
        visual_obs, flat_obs = obs
        visual_obs = U.to_float_tensor(visual_obs)
        flat_obs = U.to_float_tensor(flat_obs)
        assert torch.is_tensor(visual_obs)
        assert torch.is_tensor(flat_obs)
        if self.sleep_time > 0.0:
            time.sleep(self.sleep_time)
        visual_obs = Variable(visual_obs.unsqueeze(0))
        flat_obs = Variable(flat_obs.unsqueeze(0))
        obs = (visual_obs, flat_obs)
        action = self.model.forward_actor(obs).data.numpy()[0]
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

