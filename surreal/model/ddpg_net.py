from queue import Queue
import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np
import itertools
import torchx.nn as nnx

from .model_builders import *
from .z_filter import ZFilter

class DDPGModel(nnx.Module):

    def __init__(self,
                 obs_spec,
                 action_dim,
                 use_layernorm,
                 actor_fc_hidden_sizes,
                 critic_fc_hidden_sizes,
                 conv_out_channels,
                 conv_kernel_sizes,
                 conv_strides,
                 conv_hidden_dim,
                 critic_only=False,
                 ):
        super(DDPGModel, self).__init__()

        # hyperparameters
        self.is_pixel_input = 'pixel' in obs_spec
        self.action_dim = action_dim
        self.use_layernorm = use_layernorm

        if self.is_pixel_input:
            self.input_dim = obs_spec['pixel']['camera0']
        else:
            self.input_dim = obs_spec['low_dim']['flat_inputs'][0]

        concatenated_perception_dim = 0
        if self.is_pixel_input:
            self.perception = CNNStemNetwork(self.input_dim, conv_hidden_dim, conv_channels=conv_out_channels,
                                             kernel_sizes=conv_kernel_sizes, strides=conv_strides)
            concatenated_perception_dim += conv_hidden_dim
        if 'low_dim' in obs_spec:
            concatenated_perception_dim += obs_spec['low_dim']['flat_inputs'][0]
        if not critic_only:
            self.actor = ActorNetworkX(concatenated_perception_dim, self.action_dim, hidden_sizes=actor_fc_hidden_sizes,
                                       use_layernorm=self.use_layernorm)
        else:
            self.actor = None
        self.critic = CriticNetworkX(concatenated_perception_dim, self.action_dim, hidden_sizes=critic_fc_hidden_sizes,
                                     use_layernorm=self.use_layernorm)

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
        concatenated_inputs = []
        if self.is_pixel_input:
            obs_pixel = obs['pixel']['camera0']
            obs_pixel = self.scale_image(obs_pixel)
            cnn_updated = self.perception(obs_pixel)
            concatenated_inputs.append(cnn_updated)
        if 'low_dim' in obs:
            concatenated_inputs.append(obs['low_dim']['flat_inputs'])
        concatenated_inputs = torch.cat(concatenated_inputs, dim=1)
        return concatenated_inputs

    def forward(self, obs_in, calculate_value=True, action=None):
        obs_in = self.forward_perception(obs_in)
        if action is None:
            action = self.forward_actor(obs_in)
        value = None
        if calculate_value:
            value = self.forward_critic(obs_in, action)
        return action, value

    def scale_image(self, obs, scaling_factor=255.0):
        '''
        Given uint8 input from the environment, scale to float32 and
        divide by 255 to scale inputs between 0.0 and 1.0
        '''
        return obs / scaling_factor
        