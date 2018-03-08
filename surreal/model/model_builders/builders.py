import torch
import torch.nn as nn
import torch.nn.functional as F
import surreal.utils as U

class ActorNetwork(U.Module):

    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], conv_channels=[1,1], use_batchnorm=False):
        super(ActorNetwork, self).__init__()
        self.use_batchnorm = use_batchnorm
        D_obs_visual, D_obs_flat = D_obs
        # TODO: support both visual and flat observations
        #print('input_obs_network', D_obs_visual, D_obs_flat)
        assert not (D_obs_visual is not None and D_obs_flat is not None)
        if D_obs_visual is not None:
            # D_obs_visual should be (C, H, W)
            C, H, W = D_obs_visual.shape
            self.conv1 = nn.Conv2d(C, conv_channels[0], [3,3], stride=2)
            self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], [3,3], stride=2)
            conv_output_size = int(conv_channels[1] * H / 4 * W / 4)
            conv_output_size = 12800
            conv_output_size = 400
            #print('channels', conv_channels[1])
            #print('chw', C, H, W)
            #print('size',conv_output_size)
            self.fc_hidden = nn.Linear(conv_output_size, 50)
            self.fc_act = nn.Linear(50, D_act)
        if D_obs_flat is not None:
            self.fc_h1 = nn.Linear(D_obs, hidden_sizes[0])
            self.fc_h2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.fc_act = nn.Linear(hidden_sizes[1], D_act)

        if self.use_batchnorm:
            self.bn_h1 = nn.BatchNorm1d(D_obs)
            self.bn_h2 = nn.BatchNorm1d(hidden_sizes[0])
            self.bn_out = nn.BatchNorm1d(hidden_sizes[1])

    def forward(self, obs):
        #print('obs', type(obs))
        obs_visual, obs_flat = obs
        assert not (obs_visual is not None and obs_flat is not None)
        if obs_visual is not None:
            obs = obs_visual
            c1 = self.conv1(obs)
            c1 = F.relu(c1) # TODO: elu activation
            c2 = self.conv2(c1)
            c2 = F.relu(c2)
            batch_size = c2.shape[0]
            c2 = c2.view(batch_size, -1)
            #print(c2.shape)
            hidden = self.fc_hidden(c2)
            hidden = F.relu(hidden)
            action = self.fc_act(hidden)
            action = F.tanh(action)
            return action
        if obs_flat is not None:
            obs = obs_flat
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

    def __init__(self, D_obs, D_act, hidden_sizes=[64, 64], conv_channels=[1,1], use_batchnorm=False):
        super(CriticNetwork, self).__init__()
        self.use_batchnorm = use_batchnorm
        D_obs_visual, D_obs_flat = D_obs
        if self.use_batchnorm:
            self.bn_obs = nn.BatchNorm1d(D_obs)
            self.bn_act = nn.BatchNorm1d(D_act)
            # Critic architecture from https://github.com/Breakend/baselines/blob/50ffe01d254221db75cdb5c2ba0ab51a6da06b0a/baselines/ddpg/models.py
            self.bn_h2 = nn.BatchNorm1d(hidden_sizes[0] + D_act)
            self.bn_out = nn.BatchNorm1d(hidden_sizes[1])
        assert not (D_obs_visual is not None and D_obs_flat is not None)
        if D_obs_visual is not None:
            # D_obs_visual should be (C, H, W)
            C, H, W = D_obs_visual.shape
            self.conv1 = nn.Conv2d(C, conv_channels[0], [3,3], stride=2)
            self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], [3,3], stride=2)
            conv_output_size = conv_channels[1] * H / 4 * W / 4
            conv_output_size = 400
            #print('conv', conv_output_size)
            #print('chw', C, H, W)
            conv_output_size = int(conv_output_size)
            self.fc_hidden = nn.Linear(conv_output_size + D_act, 50)
            self.fc_act = nn.Linear(50, D_act)
        if D_obs_flat is not None:
            self.fc_obs = nn.Linear(D_obs, hidden_sizes[0])
            self.fc_h2 = nn.Linear(hidden_sizes[0] + D_act, hidden_sizes[1])
            self.fc_q = nn.Linear(hidden_sizes[1], 1)

    def forward(self, obs, act):
        obs_visual, obs_flat = obs
        assert not (obs_visual is not None and obs_flat is not None)
        if obs_visual is not None:
            obs = obs_visual
            c1 = self.conv1(obs)
            c1 = F.relu(c1) # TODO: elu activation
            c2 = self.conv2(c1)
            c2 = F.relu(c2)
            batch_size = c2.shape[0]
            c2 = c2.view(batch_size, -1)
            c2 = torch.cat((c2, act), dim=1)
            hidden = self.fc_hidden(c2)
            hidden = F.relu(hidden)
            action = self.fc_act(hidden)
            action = F.tanh(action)
            return action
        if obs_flat is not None:
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
