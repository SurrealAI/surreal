import torch
#import torch.nn as nn
#import torch.nn.functional as F
import surreal.utils as U
import torchx as tx
import torchx.nn as nnx
import torchx.layers as L
#from surreal.utils.pytorch import GpuVariable as Variable
import numpy as np 
import resource

from ..layer_norm import LayerNorm

class CNNStemNetwork(nnx.Module):
    def __init__(self, D_obs, D_out, use_layernorm=True):
        super(CNNStemNetwork, self).__init__()
        conv_channels=[16, 32]
        self.model = L.Sequential(
            L.Conv2d(conv_channels[0], kernel_size=8, stride=4),
            L.ReLU(),
            L.Conv2d(conv_channels[1], kernel_size=4, stride=2),
            L.ReLU(),
            L.Flatten(),
            L.Linear(D_out),
            L.ReLU(),
        )

        # instantiate parameters
        self.model.build(input_shape)
        print('cnnstem', list(self.parameters()))

    def forward(self, obs):
        return self.model(obs)
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 5)
        if if_high_dim: 
            obs = obs.view(-1, *obs_shape[2:])

        obs = Fx.relu(self.conv1(obs))
        obs = Fx.relu(self.conv2(obs))
        obs = obs.view(obs.size(0), -1)
        obs = Fx.relu(self.fc_obs(obs))

        if if_high_dim:
            obs = obs.view(obs_shape[0], obs_shape[1], -1)
        return obs

class ActorNetworkX(nnx.Module):
    def __init__(self, D_in, D_act, hidden_size=200, use_layernorm=True):
        super(ActorNetworkX, self).__init__()
        self.fc_in = L.Linear(hidden_size)
        self.fc_out = L.Linear(D_act)
        self.relu = L.ReLU()
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.layer_norm = LayerNorm()
        print('actor', list(self.parameters()))

    def forward(self, obs):
        x = self.relu(self.fc_in(obs))
        if self.use_layernorm:
            x = self.layer_norm(x)
        x = self.tanh(self.fc_out(x))
        return x

class CriticNetworkX(nnx.Module):
    def __init__(self, D_in, D_act, hidden_size=300, use_layernorm=True):
        super(CriticNetworkX, self).__init__()
        self.fc_in = L.Linear(hidden_size)
        self.fc_out = L.Linear(1)
        self.relu = L.ReLU()
        self.flatten = L.Flatten()
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.layer_norm = LayerNorm()
        print('critic', list(self.parameters()))

        #model.build() # model is after the concat

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=1)
        x = self.relu(self.fc_in(x))
        if self.use_layernorm:
            x = self.layer_norm(x)
        x = self.fc_out(x)
        return x

'''
class ActorNetwork(U.Module):
    \'''
    For use with flat observations
    \'''

    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], use_batchnorm=False):
        super(ActorNetwork, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.fc_h1 = nn.Linear(D_obs, hidden_sizes[0])
        self.fc_h2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_act = nn.Linear(hidden_sizes[1], D_act)

        if self.use_batchnorm:
            self.bn_h1 = nn.BatchNorm1d(D_obs)
            self.bn_h2 = nn.BatchNorm1d(hidden_sizes[0])
            self.bn_out = nn.BatchNorm1d(hidden_sizes[1])

    def forward(self, obs):
        if self.use_batchnorm:
            obs = self.bn_h1(obs)
        h1 = F.relu(self.fc_h1(obs))
        if self.use_batchnorm:
            h1 = self.bn_h2(h1)
        h2 = F.relu(self.fc_h2(h1))
        if self.use_batchnorm:
            h2 = self.bn_out(h2)
        action = F.tanh(self.fc_act(h2))
        return action

class CriticNetwork(nnx.Module):

    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], use_batchnorm=False):
        super(CriticNetwork, self).__init__()
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn_obs = nn.BatchNorm1d(D_obs)
            self.bn_act = nn.BatchNorm1d(D_act)
            # Critic architecture from https://github.com/Breakend/baselines/blob/50ffe01d254221db75cdb5c2ba0ab51a6da06b0a/baselines/ddpg/models.py
            self.bn_h2 = nn.BatchNorm1d(hidden_sizes[0] + D_act)
            self.bn_out = nn.BatchNorm1d(hidden_sizes[1])
        self.fc_obs = nn.Linear(D_obs, hidden_sizes[0])
        self.fc_h2 = nn.Linear(hidden_sizes[0] + D_act, hidden_sizes[1])
        self.fc_q = nn.Linear(hidden_sizes[1], 1)

    def forward(self, obs, act):
        if self.use_batchnorm:
            obs = self.bn_obs(obs)
        h_obs = F.relu(self.fc_obs(obs))
        h1 = torch.cat((h_obs, act), 1)
        if self.use_batchnorm:
            h1 = self.bn_h2(h1)
        h2 = F.relu(self.fc_h2(h1))
        if self.use_batchnorm:
            h2 = self.bn_out(h2)
        value = self.fc_q(h2)
        return value

class PPO_ActorNetwork(nnx.Module):
    \'''
        PPO custom actor network structure
    \'''
    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], init_log_sig=0):
        super(PPO_ActorNetwork, self).__init__()
        # assumes D_obs here is the correct RNN hidden dim

        self.fc_h1 = nn.Linear(D_obs, hidden_sizes[0])
        self.fc_h2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_mean = nn.Linear(hidden_sizes[1], D_act)
        self.log_var = nn.Parameter(torch.zeros(1, D_act) + init_log_sig)

    def forward(self, obs):
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 3)
        if if_high_dim: 
            obs = obs.view(-1, obs_shape[2])

        h1 = F.tanh(self.fc_h1(obs))
        h2 = F.tanh(self.fc_h2(h1))
        mean = self.fc_mean(h2)
        std  = torch.exp(self.log_var) * Variable(torch.ones(mean.size()))

        action = torch.cat((mean, std), dim=1)
        if if_high_dim:
            action = action.view(obs_shape[0], obs_shape[1], -1)
        return action


class PPO_CriticNetwork(nnx.Module):
    \'''
        PPO custom critic network structure
    \'''
    def __init__(self, D_obs, hidden_sizes=[64, 64]):
        super(PPO_CriticNetwork, self).__init__()
        # assumes D_obs here is the correct RNN hidden dim if necessary

        self.fc_h1 = nn.Linear(D_obs, hidden_sizes[0])
        self.fc_h2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_v  = nn.Linear(hidden_sizes[1], 1)

    def forward(self, obs):
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 3)
        if if_high_dim: 
            obs = obs.view(-1, obs_shape[2])

        h1 = F.tanh(self.fc_h1(obs))
        h2 = F.tanh(self.fc_h2(h1))
        v  = self.fc_v(h2) 

        if if_high_dim:
            v = v.view(obs_shape[0], obs_shape[1], 1)
        return v

'''
