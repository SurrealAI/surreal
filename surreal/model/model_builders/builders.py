import torch
import torchx as tx
import torchx.nn as nnx
import torch.nn as nn
import torchx.layers as L
import numpy as np

class CNNStemNetwork(nnx.Module):
    def __init__(self, D_obs, D_out, conv_channels=[16, 32], kernel_sizes=[8, 4], strides=[4,2]):
        super(CNNStemNetwork, self).__init__()
        layers = []
        for i in range(len(conv_channels)):
            layers.append(L.Conv2d(conv_channels[i], kernel_size=kernel_sizes[i], stride=strides[i]))
            layers.append(L.ReLU())
        layers.append(L.Flatten())
        layers.append(L.Linear(D_out))
        layers.append(L.ReLU())
        self.model = L.Sequential(*layers)

        # instantiate parameters
        self.model.build((None, *D_obs))

    def forward(self, obs):
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 5)
        if if_high_dim: # case of RNN input
            obs = obs.view(-1, *obs_shape[2:])  

        obs = self.model(obs)

        if if_high_dim:
            obs = obs.view(obs_shape[0], obs_shape[1], -1)
        return obs

class ActorNetworkX(nnx.Module):
    def __init__(self, D_in, D_act, hidden_sizes=[300, 200], use_layernorm=True):
        super(ActorNetworkX, self).__init__()

        xp_input = L.Placeholder((None, D_in))
        xp = L.Linear(hidden_sizes[0])(xp_input)
        xp = L.ReLU()(xp)
        if use_layernorm:
            # Normalize 1 dimension
            xp = L.LayerNorm(1)(xp)
        xp = L.Linear(hidden_sizes[1])(xp)
        xp = L.ReLU()(xp)
        if use_layernorm:
            xp = L.LayerNorm(1)(xp)
        xp = L.Linear(D_act)(xp)
        xp = L.Tanh()(xp)

        self.model = L.Functional(inputs=xp_input, outputs=xp)
        self.model.build((None, D_in))

    def forward(self, obs):
        return self.model(obs)

class CriticNetworkX(nnx.Module):
    def __init__(self, D_in, D_act, hidden_sizes=[400, 300], use_layernorm=True):
        super(CriticNetworkX, self).__init__()

        xp_input_obs = L.Placeholder((None, D_in))
        xp = L.Linear(hidden_sizes[0])(xp_input_obs)
        xp = L.ReLU()(xp)
        if use_layernorm:
            xp = L.LayerNorm(1)(xp)
        self.model_obs = L.Functional(inputs=xp_input_obs, outputs=xp)
        self.model_obs.build((None, D_in))

        xp_input_concat = L.Placeholder((None, hidden_sizes[0] + D_act))
        xp = L.Linear(hidden_sizes[1])(xp_input_concat)
        xp = L.ReLU()(xp)
        if use_layernorm:
            xp = L.LayerNorm(1)(xp)
        xp = L.Linear(1)(xp)

        self.model_concat = L.Functional(inputs=xp_input_concat, outputs=xp)
        self.model_concat.build((None, D_act + hidden_sizes[0]))

    def forward(self, obs, act):
        h_obs = self.model_obs(obs)
        h1 = torch.cat((h_obs, act), 1)
        value = self.model_concat(h1)
        return value

class PPO_ActorNetwork(nnx.Module):
    '''
        PPO custom actor network structure
    '''
    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], init_log_sig=0):
        '''
            Constructor for PPO actor network
            Args: 
                D_obs: observation space dimension, scalar
                D_act: action space dimension, scalar
                hidden_sizes: list of fully connected dimension
                init_log_sig: initial value for log standard deviation parameter
        '''
        super(PPO_ActorNetwork, self).__init__()
        # assumes D_obs here is the correct RNN hidden dim
        xp_input = L.Placeholder((None, D_obs))
        xp = L.Linear(hidden_sizes[0])(xp_input)
        xp = L.ReLU()(xp)
        xp = L.Linear(hidden_sizes[1])(xp)
        xp = L.ReLU()(xp)
        xp = L.Linear(D_act)(xp)
        xp = L.Tanh()(xp)

        self.model = L.Functional(inputs=xp_input, outputs=xp)
        self.model.build((None, D_obs))

        self.log_var = nn.Parameter(torch.zeros(1, D_act) + init_log_sig)

    def forward(self, obs):
        '''
            Forward pass of actor network. Input assume to be output of CNN
            and/or LSTM feature extractor
            Args:
                obs: batched tensor denotes the states
        '''
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 3)
        if if_high_dim: 
            obs = obs.view(-1, obs_shape[2])

        mean = self.model(obs)
        std  = torch.exp(self.log_var) * torch.ones(mean.size())

        action = torch.cat((mean, std), dim=1)
        if if_high_dim:
            action = action.view(obs_shape[0], obs_shape[1], -1)
        return action


class PPO_CriticNetwork(nnx.Module):
    '''
        PPO custom critic network structure
    '''
    def __init__(self, D_obs, hidden_sizes=[64, 64]):
        '''
            Constructor for PPO critic network
            Args: 
                D_obs: observation space dimension, scalar
                hidden_sizes: list of fully connected dimension
        '''
        super(PPO_CriticNetwork, self).__init__()
        # assumes D_obs here is the correct RNN hidden dim if necessary

        xp_input = L.Placeholder((None, D_obs))
        xp = L.Linear(hidden_sizes[0])(xp_input)
        xp = L.ReLU()(xp)
        xp = L.Linear(hidden_sizes[1])(xp)
        xp = L.ReLU()(xp)
        xp = L.Linear(1)(xp)

        self.model = L.Functional(inputs=xp_input, outputs=xp)
        self.model.build((None, D_obs))

    def forward(self, obs):
        '''
            Forward pass of actor network. Input assume to be output of CNN
            and/or LSTM feature extractor
            Args:
                obs: batched tensor denotes the states
        '''
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 3)
        if if_high_dim: 
            obs = obs.view(-1, obs_shape[2])

        v = self.model(obs)

        if if_high_dim:
            v = v.view(obs_shape[0], obs_shape[1], 1)
        return v

