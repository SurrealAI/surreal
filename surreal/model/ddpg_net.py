from queue import Queue
import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np

from .model_builders import *
from .z_filter import ZFilter

class DDPGModel(U.Module):

    def __init__(self,
                 obs_spec,
                 action_dim,
                 is_uint8_pixel_input,
                 is_pixel_input,
                 use_z_filter,
                 use_batchnorm,
                 use_layernorm,
                 actor_fc_hidden_sizes,
                 critic_fc_hidden_sizes,
                 use_cuda = False):
        super(DDPGModel, self).__init__()

        # hyperparameters
        self.action_dim = action_dim
        self.is_uint8_pixel_input = is_uint8_pixel_input
        self.is_pixel_input = is_pixel_input
        self.use_z_filter = use_z_filter
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm

        print(obs_spec)
        if self.is_pixel_input:
            self.obs_spec = obs_spec['pixel']['pixels']
        else:
            self.obs_spec = obs_spec['low_dim']['flat_inputs']

        if self.is_pixel_input:
            perception_hidden_dim = 200
            self.perception = PerceptionNetwork(self.obs_spec, perception_hidden_dim,
                                                use_layernorm=self.use_layernorm)
        else:
            perception_hidden_dim = obs_spec
        self.actor = ActorNetworkX(perception_hidden_dim, self.action_dim)
        self.critic = CriticNetworkX(perception_hidden_dim, self.action_dim)

    def forward_actor(self, obs):
        return self.actor(obs)

    def forward_critic(self, obs, action):
        return self.critic(obs, action)

    def forward_perception(self, obs):
        obs = obs['pixel']['pixels']
        if self.is_uint8_pixel_input and self.is_pixel_input:
            obs = self.scale_image(obs)
        return self.perception(obs)

    def forward(self, obs_in):
        if self.is_pixel_input:
            obs_in = self.forward_perception(obs_in)
        else:
            obs_in = obs_in['low_dim']['flat_inputs']
        action = self.forward_actor(obs_in)
        value = self.forward_critic(obs, action)
        return (action, value)

    def scale_image(self, obs):
        '''
        Given uint8 input from the environment, scale to float32 and
        divide by 256 to scale inputs between 0.0 and 1.0
        '''
        return obs / 256.0
        