import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np

from .model_builders import *


class DDPGModel(U.Module):

    def __init__(self,
                 obs_dim,
                 action_dim):
        super(DDPGModel, self).__init__()

        # hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = ActorNetwork(self.obs_dim, self.action_dim)
        self.critic = CriticNetwork(self.obs_dim, self.action_dim)

    def forward_actor(self, obs):
        shape = obs.size()
        assert len(shape) == 2 and shape[1] == self.obs_dim
        return self.actor(obs)

    def forward_critic(self, obs, action):
        obs_shape = obs.size()
        assert len(obs_shape) == 2 and obs_shape[1] == self.obs_dim
        action_shape = action.size()
        assert len(action_shape) == 2 and action_shape[1] == self.action_dim
        return self.critic(obs, action)

    def forward(self, obs):
        action = self.forward_actor(obs)
        value = self.forward_critic(obs, action)
        return (action, value)
