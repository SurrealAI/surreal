import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np

from .model_builders import *
from .z_filter import ZFilter

class DDPGModel(U.Module):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 use_z_filter,
                 use_batchnorm,
                 actor_fc_hidden_sizes,
                 critic_fc_hidden_sizes,):
        super(DDPGModel, self).__init__()

        # hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_z_filter = use_z_filter
        self.use_batchnorm = use_batchnorm

        self.actor = ActorNetwork(self.obs_dim, self.action_dim, use_batchnorm=use_batchnorm, hidden_sizes=actor_fc_hidden_sizes)
        self.critic = CriticNetwork(self.obs_dim, self.action_dim, use_batchnorm=use_batchnorm, hidden_sizes=critic_fc_hidden_sizes)
        if self.use_z_filter:
            self.z_filter = ZFilter(obs_dim)

    def forward_actor(self, obs):
        shape = obs.size()
        assert len(shape) == 2 and shape[1] == self.obs_dim
        if self.use_z_filter:
            obs = self.z_filter.forward(obs)
        return self.actor(obs)

    def forward_critic(self, obs, action):
        obs_shape = obs.size()
        assert len(obs_shape) == 2 and obs_shape[1] == self.obs_dim
        action_shape = action.size()
        assert len(action_shape) == 2 and action_shape[1] == self.action_dim
        if self.use_z_filter:
            obs = self.z_filter.forward(obs)
        return self.critic(obs, action)

    def forward(self, obs):
        action = self.forward_actor(obs)
        value = self.forward_critic(obs, action)
        return (action, value)

    def z_update(self, obs):
        if self.use_z_filter:
            self.z_filter.z_update(obs)
        else:
            raise ValueError('Z_update called when network is set to not use z_filter')
