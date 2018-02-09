import torch
import torch.nn as nn
import torch.nn.functional as F
import surreal.utils as U

class ActorNetwork(U.Module):

    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], use_batchnorm=False, train=False):
        super(ActorNetwork, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.fc_h1 = nn.Linear(D_obs, hidden_sizes[0])
        self.fc_h2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_act = nn.Linear(hidden_sizes[1], D_act)

        if self.use_batchnorm:
            self.bn_h1 = nn.BatchNorm1d(D_obs)
            self.bn_h2 = nn.BatchNorm1d(hidden_sizes[0])
            self.bn_out = nn.BatchNorm1d(hidden_sizes[1])
            if not train:
                self.bn_h1.eval()
                self.bn_h2.eval()
                self.bn_out.eval()

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

    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], use_batchnorm=False, train=False):
        super(CriticNetwork, self).__init__()
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn_obs = nn.BatchNorm1d(D_obs)
            self.bn_act = nn.BatchNorm1d(D_act)
            # Critic architecture from https://github.com/Breakend/baselines/blob/50ffe01d254221db75cdb5c2ba0ab51a6da06b0a/baselines/ddpg/models.py
            self.bn_h2 = nn.BatchNorm1d(hidden_sizes[0] + D_act)
            self.bn_out = nn.BatchNorm1d(hidden_sizes[1])
            if not train:
                self.bn_obs.eval()
                self.bn_act.eval()
                self.bn_h2.eval()
                self.bn_out.eval()
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
