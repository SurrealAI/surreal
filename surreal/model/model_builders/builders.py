import torch
import torch.nn as nn
import torch.nn.functional as F
import surreal.utils as U
from surreal.utils.pytorch import GpuVariable as Variable
import numpy as np 

class ActorNetwork(U.Module):

    def __init__(self, D_obs, D_act):
        super(ActorNetwork, self).__init__()
        self.fc_h1 = nn.Linear(D_obs, 64)
        self.fc_h2 = nn.Linear(64, 64)
        self.fc_act = nn.Linear(64, D_act)

    def forward(self, obs):
        h1 = F.relu(self.fc_h1(obs))
        h2 = F.relu(self.fc_h2(h1))
        action = F.tanh(self.fc_act(h2))
        return action


class CriticNetwork(U.Module):

    def __init__(self, D_obs, D_act):
        super(CriticNetwork, self).__init__()
        self.fc_obs = nn.Linear(D_obs, 32)
        self.fc_act = nn.Linear(D_act, 32)
        # self.fc_h1 = nn.Linear(D_obs+D_act, 128)
        self.fc_h2 = nn.Linear(64, 64)
        self.fc_q = nn.Linear(64, 1)

    def forward(self, obs, act):
        h_obs = F.relu(self.fc_obs(obs))
        h_act = F.relu(self.fc_act(act))
        # concat_input = torch.cat((obs, act), 1)
        # h1 = F.relu(self.fc_h1(concat_input))
        h1 = torch.cat((h_obs, h_act), 1)
        h2 = F.relu(self.fc_h2(h1))
        value = self.fc_q(h2)
        return value


class PPO_ActorNetwork(U.Module):

    def __init__(self, D_obs, D_act, init_log_sig):
        super(PPO_ActorNetwork, self).__init__()
        hid_1 = D_obs * 10
        hid_3 = D_act * 10
        hid_2 = int(np.sqrt(hid_1 * hid_3))
        self.fc_h1 = nn.Linear(D_obs, hid_1)
        self.fc_h2 = nn.Linear(hid_1, hid_2)
        self.fc_h3 = nn.Linear(hid_2, hid_3)
        self.fc_mean = nn.Linear(hid_3, D_act)
        self.log_var = nn.Parameter(torch.zeros(1, D_act) + init_log_sig)

    def forward(self, obs):
        h1 = F.tanh(self.fc_h1(obs))
        h2 = F.tanh(self.fc_h2(h1))
        h3 = F.tanh(self.fc_h3(h2))
        mean = self.fc_mean(h3)
        std  = torch.exp(self.log_var * 0.5) * Variable(torch.ones(mean.size()))
        action = torch.cat((mean, std), dim=1)
        return action


class PPO_CriticNetwork(U.Module):

    def __init__(self, D_obs):
        super(PPO_CriticNetwork, self).__init__()
        hid_1 = D_obs * 10
        hid_3 = 5
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
