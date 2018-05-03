from queue import Queue
import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np
import itertools

from .model_builders import *
from .z_filter import ZFilter

class DDPGModel(U.Module):

    def __init__(self,
                 obs_spec,
                 action_dim,
                 use_layernorm,
                 actor_fc_hidden_sizes,
                 critic_fc_hidden_sizes,
                 use_z_filter=False,
                 ):
        super(DDPGModel, self).__init__()

        # hyperparameters
        self.is_pixel_input = 'pixel' in obs_spec
        self.action_dim = action_dim
        self.use_z_filter = use_z_filter
        self.use_layernorm = use_layernorm

        if self.is_pixel_input:
            self.input_dim = obs_spec['pixel']['camera0']
        else:
            self.input_dim = obs_spec['low_dim']['flat_inputs'][0]

        if self.is_pixel_input:
            perception_hidden_dim = 200
            self.perception = CNNStemNetwork(self.input_dim, perception_hidden_dim)
            self.actor = ActorNetworkX(perception_hidden_dim, self.action_dim, use_layernorm=self.use_layernorm)
            self.critic = CriticNetworkX(perception_hidden_dim, self.action_dim, use_layernorm=self.use_layernorm)
        else:
            self.actor = ActorNetwork(self.input_dim, self.action_dim, hidden_sizes=actor_fc_hidden_sizes)
            self.critic = CriticNetwork(self.input_dim, self.action_dim, hidden_sizes=critic_fc_hidden_sizes)

    def get_actor_parameters(self):
        return itertools.chain(self.actor.parameters())

    def get_critic_parameters(self):
        if self.is_pixel_input:
            return itertools.chain(self.critic.parameters(), self.perception.parameters())
        else:
            return itertools.chain(self.critic.parameters())

    def forward_actor(self, obs):
        return self.actor(obs)

    def forward_critic(self, obs, action):
        return self.critic(obs, action)

    def forward_perception(self, obs):
        if self.is_pixel_input:
            obs = obs['pixel']['camera0']
            obs = self.scale_image(obs)
            return self.perception(obs)
        else:
            obs = obs['low_dim']['flat_inputs']
            return obs

    def forward(self, obs_in, calculate_value=True):
        obs_in = self.forward_perception(obs_in)
        action = self.forward_actor(obs_in)
        value = None
        if calculate_value:
            value = self.forward_critic(obs_in, action)
        return action, value

    def scale_image(self, obs):
        '''
        Given uint8 input from the environment, scale to float32 and
        divide by 256 to scale inputs between 0.0 and 1.0
        '''
        return obs / 256.0
        