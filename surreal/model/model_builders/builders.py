import surreal.utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):

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


class CriticNetwork(nn.Module):

    def __init__(self, D_obs, D_act):
        super(CriticNetwork, self).__init__()
        self.fc_obs = nn.Linear(D_obs, 32)
        self.fc_act = nn.Linear(D_act, 32)
        self.fc_h2 = nn.Linear(64, 64)
        self.fc_q = nn.Linear(64, 1)

    def forward(self, obs, act):
        h_obs = F.relu(self.fc_obs(obs))
        h_act = F.relu(self.fc_act(act))
        h1 = torch.cat((h_obs, h_act), 1)
        h2 = F.relu(self.fc_h2(h1))
        value = self.fc_q(h2)
        return value
