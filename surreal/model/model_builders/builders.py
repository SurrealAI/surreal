import torch
import torch.nn as nn
import torch.nn.functional as F
import surreal.utils as U
from surreal.utils.pytorch import GpuVariable as Variable
import numpy as np 
import resource

class ActorNetwork(U.Module):

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

class CriticNetwork(U.Module):

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


class PPO_ActorNetwork(U.Module):
    '''
        PPO custom actor network structure
    '''
    def __init__(self, D_obs, D_act, init_log_sig, rnn_stem=None):
        super(PPO_ActorNetwork, self).__init__()

        self.rnn_stem = rnn_stem
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

    def forward(self, obs):
        print('\tCheckpoint 2.4.1: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        h1 = F.tanh(self.fc_h1(obs))
        h2 = F.tanh(self.fc_h2(h1))
        h3 = F.tanh(self.fc_h3(h2))
        print('\tCheckpoint 2.4.2: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        mean = self.fc_mean(h3)
        std  = torch.exp(self.log_var) * Variable(torch.ones(mean.size()))
        print('\tCheckpoint 2.4.3: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        action = torch.cat((mean, std), dim=1)
        print('\tCheckpoint 2.4.4: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return action


class PPO_CriticNetwork(U.Module):
    '''
        PPO custom critic network structure
    '''
    def __init__(self, D_obs, rnn_stem=None):
        super(PPO_CriticNetwork, self).__init__()

        # assumes D_obs here is the correct RNN hidden dim
        self.rnn_stem = rnn_stem

        hid_1 = D_obs * 10
        hid_3 = 64
        hid_2 = int(np.sqrt(hid_1 * hid_3))

        self.fc_h1 = nn.Linear(D_obs, hid_1)
        self.fc_h2 = nn.Linear(hid_1, hid_2)
        self.fc_h3 = nn.Linear(hid_2, hid_3)
        self.fc_v  = nn.Linear(hid_3, 1)

    def forward(self, obs):
        h1 = F.tanh(self.fc_h1(obs))
        h2 = F.tanh(self.fc_h2(h1))
        h3 = F.tanh(self.fc_h3(h2))
        v  = self.fc_v(h3) 
        return v

