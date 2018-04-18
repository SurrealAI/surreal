import torch
import torch.nn as nn
import torch.nn.functional as F
import surreal.utils as U
from surreal.utils.pytorch import GpuVariable as Variable
import numpy as np 
import resource

from ..layer_norm import LayerNorm

class PerceptionNetwork(U.Module):
    def __init__(self, D_obs, D_out, use_layernorm=True):
        super(PerceptionNetwork, self).__init__()
        self.use_layernorm = use_layernorm
        D_obs, _ = D_obs # Unpacking D_obs_visual, D_obs_flat, TODO: fix this strange line
        if use_layernorm:
            self.layer_norm = LayerNorm()
        conv_channels=[32, 32]
        C, H, W = D_obs.shape
        self.conv1 = nn.Conv2d(C, 32, [3,3], stride=2)
        self.conv2 = nn.Conv2d(32, 32, [3,3], stride=1)
        # TODO: auto shape inference
        conv_output_size = 48672
        self.fc_obs = nn.Linear(conv_output_size, D_out)

    def forward(self, obs_in):
        obs, _ = obs_in # Unpacking obs_visual, obs_flat, TODO: fix this strange line
        obs = F.elu(self.conv1(obs))
        obs = F.elu(self.conv2(obs))
        obs = obs.view(obs.size(0), -1)
        obs = F.elu(self.fc_obs(obs))
        return obs

class ActorNetworkX(U.Module):
    def __init__(self, D_in, D_act, hidden_size=200):
        super(ActorNetworkX, self).__init__()
        self.fc_in = nn.Linear(D_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, D_act)
        self.layer_norm = LayerNorm()

    def forward(self, obs):
        x = F.elu(self.fc_in(obs))
        x = self.layer_norm(x)
        x = F.tanh(self.fc_out(x))
        return x
        
class CriticNetworkX(U.Module):
    def __init__(self, D_in, D_act, hidden_size=300):
        super(CriticNetworkX, self).__init__()
        self.fc_in = nn.Linear(D_in + D_act, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.layer_norm = LayerNorm()

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=1)
        x = F.elu(self.fc_in(x))
        x = self.layer_norm(x)
        x = self.fc_out(x)
        return x

class ActorNetwork(U.Module):

    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], conv_channels=[32, 32], use_batchnorm=False, use_layernorm=True):
        super(ActorNetwork, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        D_obs_visual, D_obs_flat = D_obs
        # TODO: support both visual and flat observations
        #print('input_obs_network', D_obs_visual, D_obs_flat)
        assert not (D_obs_visual is not None and D_obs_flat is not None)
        if D_obs_visual is not None:
            if use_layernorm:
                self.layer_norm = LayerNorm()
            # D_obs_visual should be (C, H, W)
            C, H, W = D_obs_visual.shape
            self.conv1 = nn.Conv2d(C, conv_channels[0], [3,3], stride=2)
            self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], [3,3], stride=1)
            #conv_output_size = int(conv_channels[1] * H / 4 * W / 4)
            conv_output_size = 400 * 32
            conv_output_size = 48672
            #print('channels', conv_channels[1])
            #print('chw', C, H, W)
            #print('size',conv_output_size)
            self.fc_obs = nn.Linear(conv_output_size, 50)
            self.fc_hidden = nn.Linear(50, 50)
            self.fc_act = nn.Linear(50, D_act)
        if D_obs_flat is not None:
            self.fc_h1 = nn.Linear(D_obs_flat, hidden_sizes[0])
            self.fc_h2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.fc_act = nn.Linear(hidden_sizes[1], D_act)

        if self.use_batchnorm:
            self.bn_h1 = nn.BatchNorm1d(D_obs_flat)
            self.bn_h2 = nn.BatchNorm1d(hidden_sizes[0])
            self.bn_out = nn.BatchNorm1d(hidden_sizes[1])

    def forward(self, obs):
        obs_visual, obs_flat = obs
        assert not ((obs_visual is not None) and (obs_flat is not None))
        if obs_visual is not None:
            obs = obs_visual
            c1 = self.conv1(obs)
            c1 = F.elu(c1)
            c2 = self.conv2(c1)
            c2 = F.elu(c2)
            batch_size = c2.size()[0]
            c2 = c2.view(batch_size, -1)
            flat_obs = self.fc_obs(c2)
            if self.use_layernorm:
                flat_obs = self.layer_norm(flat_obs)
            flat_obs = F.elu(flat_obs)
            hidden = self.fc_hidden(flat_obs)
            hidden = F.elu(hidden)
            action = self.fc_act(hidden)
            action = F.tanh(action)
            return action
        if obs_flat is not None:
            obs = obs_flat
            if self.use_batchnorm:
                obs = self.bn_h1(obs)
            h1 = F.elu(self.fc_h1(obs))
            if self.use_batchnorm:
                h1 = self.bn_h2(h1)
            h2 = F.elu(self.fc_h2(h1))
            if self.use_batchnorm:
                h2 = self.bn_out(h2)
            action = F.tanh(self.fc_act(h2))
            return action

class CriticNetwork(U.Module):

    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], conv_channels=[32, 32], use_batchnorm=False, use_layernorm=True):
        super(CriticNetwork, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        D_obs_visual, D_obs_flat = D_obs
        if self.use_batchnorm:
            self.bn_obs = nn.BatchNorm1d(D_obs_flat)
            self.bn_act = nn.BatchNorm1d(D_act)
            # Critic architecture from https://github.com/Breakend/baselines/blob/50ffe01d254221db75cdb5c2ba0ab51a6da06b0a/baselines/ddpg/models.py
            self.bn_h2 = nn.BatchNorm1d(hidden_sizes[0] + D_act)
            self.bn_out = nn.BatchNorm1d(hidden_sizes[1])
        assert not (D_obs_visual is not None and D_obs_flat is not None)
        if D_obs_visual is not None:
            if use_layernorm:
                self.layer_norm = LayerNorm()
            # D_obs_visual should be (C, H, W)
            C, H, W = D_obs_visual.shape
            self.conv1 = nn.Conv2d(C, conv_channels[0], [3,3], stride=2)
            self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], [3,3], stride=1)
            #conv_output_size = conv_channels[1] * H / 4 * W / 4
            conv_output_size = 400 * 32
            conv_output_size = 48672
            #print('conv', conv_output_size)
            #print('chw', C, H, W)
            conv_output_size = int(conv_output_size)
            self.fc_obs = nn.Linear(conv_output_size, 50)
            # TODO: consider 1 mlp instead of 2, or different hidden size
            self.fc_hidden = nn.Linear(50 + D_act, 50)
            self.fc_out = nn.Linear(50, D_act)
        if D_obs_flat is not None:
            self.fc_obs = nn.Linear(D_obs_flat, hidden_sizes[0])
            self.fc_h2 = nn.Linear(hidden_sizes[0] + D_act, hidden_sizes[1])
            self.fc_q = nn.Linear(hidden_sizes[1], 1)

    def forward(self, obs, act):
        obs_visual, obs_flat = obs
        assert not (obs_visual is not None and obs_flat is not None)
        if obs_visual is not None:
            obs = obs_visual
            c1 = self.conv1(obs)
            c1 = F.elu(c1)
            c2 = self.conv2(c1)
            c2 = F.elu(c2)
            batch_size = c2.size()[0]
            c2 = c2.view(batch_size, -1)
            flat_obs = self.fc_obs(c2)
            if self.use_layernorm:
                flat_obs = self.layer_norm(flat_obs)
            flat_obs = F.elu(flat_obs)
            flat_obs = torch.cat((flat_obs, act), dim=1)
            hidden = self.fc_hidden(flat_obs)
            hidden = F.elu(hidden)
            value = self.fc_out(hidden)
            return value
        if obs_flat is not None:
            obs = obs_flat
            if self.use_batchnorm:
                obs = self.bn_obs(obs)
            h_obs = F.elu(self.fc_obs(obs))
            h1 = torch.cat((h_obs, act), 1)
            if self.use_batchnorm:
                h1 = self.bn_h2(h1)
            h2 = F.elu(self.fc_h2(h1))
            if self.use_batchnorm:
                h2 = self.bn_out(h2)
            value = self.fc_q(h2)
            return value

class PPO_ActorNetwork(U.Module):
    '''
        PPO custom actor network structure
    '''
    def __init__(self, D_obs, D_act, init_log_sig, cnn_stem=None, rnn_stem=None):
        super(PPO_ActorNetwork, self).__init__()

        self.rnn_stem = rnn_stem
        self.cnn_stem = cnn_stem
        # assumes D_obs here is the correct RNN hidden dim

        self.D_obs = D_obs
        hid_1 = D_obs * 10
        hid_3 = D_act * 10
        hid_2 = int(np.sqrt(hid_1 * hid_3))
        self.fc_h1 = nn.Linear(D_obs, hid_1)
        self.fc_h2 = nn.Linear(hid_1, hid_2)
        self.fc_h3 = nn.Linear(hid_2, hid_3)
        self.fc_mean = nn.Linear(hid_3, D_act)
        self.log_var = nn.Parameter(torch.zeros(1, D_act) + init_log_sig)
        # self.layer_norm = LayerNorm()

    def forward(self, obs):
        h1 = F.tanh(self.fc_h1(obs))
        h2 = F.tanh(self.fc_h2(h1))
        h3 = F.tanh(self.fc_h3(h2))
        mean = self.fc_mean(h3)
        std  = torch.exp(self.log_var) * Variable(torch.ones(mean.size()))

        action = torch.cat((mean, std), dim=1)
        return action


class PPO_CriticNetwork(U.Module):
    '''
        PPO custom critic network structure
    '''
    def __init__(self, D_obs, cnn_stem=None,rnn_stem=None):
        super(PPO_CriticNetwork, self).__init__()

        # assumes D_obs here is the correct RNN hidden dim if necessary
        self.rnn_stem = rnn_stem
        self.cnn_stem = cnn_stem

        hid_1 = D_obs * 10
        hid_3 = 64
        hid_2 = int(np.sqrt(hid_1 * hid_3))

        self.fc_h1 = nn.Linear(D_obs, hid_1)
        self.fc_h2 = nn.Linear(hid_1, hid_2)
        self.fc_h3 = nn.Linear(hid_2, hid_3)
        self.fc_v  = nn.Linear(hid_3, 1)
        # self.layer_norm = LayerNorm()

    def forward(self, obs):
        h1 = F.tanh(self.fc_h1(obs))
        h2 = F.tanh(self.fc_h2(h1))
        h3 = F.tanh(self.fc_h3(h2))
        v  = self.fc_v(h3) 
        return v

