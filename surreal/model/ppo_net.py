import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np

from .model_builders import *
from .z_filter import ZFilter

class DiagGauss(object):
    def __init__(self, action_dim):
        self.d = action_dim

    def loglikelihood(self, a, prob):
        mean0 = prob[:, :self.d]
        std0 = prob[:, self.d:]
        return - 0.5 * (((a - mean0) / std0).pow(2)).sum(dim=1, keepdim=True) - 0.5 * np.log(
            2.0 * np.pi) * self.d - std0.log().sum(dim=1, keepdim=True)

    def likelihood(self, a, prob):
        return torch.clamp(self.loglikelihood(a, prob).exp(), min=1e-5)

    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return ((std1 / std0).log()).sum(dim=1) + (
            (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2))).sum(dim=1) - 0.5 * self.d

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return std_nd.log().sum(dim=1) + .5 * np.log(2 * np.pi * np.e) * self.d

    def sample(self, prob):
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]


class PPOModel(U.Module):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 use_z_filter):
        super(PPOModel, self).__init__()

        # hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_z_filter = use_z_filter

        self.actor = PPO_ActorNetwork(self.obs_dim, self.action_dim)
        self.critic = PPO_CriticNetwork(self.obs_dim)
        if self.use_z_filter:
            self.z_filter = ZFilter(obs_dim)

    def forward(self, obs):
        shape = obs.size()
        assert len(shape) == 2 and shape[1] == self.obs_dim
        if self.use_z_filter:
            obs = self.z_filter.forward(obs)

        action = self.actor(obs)
        value  = self.critic(obs)
        return (action, value)

    def z_update(self, obs):
        if self.use_z_filter:
            self.z_filter.z_update(obs)
        else:
            raise ValueError('Z_update called when network is set to not use z_filter')
