import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np

from .model_builders import *
from .z_filter import ZFilter

class DDPGModel(U.Module):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 is_uint8_pixel_input,
                 use_z_filter,
                 use_batchnorm,
                 use_layernorm,
                 actor_fc_hidden_sizes,
                 critic_fc_hidden_sizes,
                 use_cuda = False):
        super(DDPGModel, self).__init__()

        # hyperparameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.is_uint8_pixel_input = is_uint8_pixel_input
        self.use_z_filter = use_z_filter
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm

        perception_hidden_dim = 50
        self.perception = PerceptionNetwork(self.obs_dim, perception_hidden_dim,
                                            use_layernorm=self.use_layernorm)
        self.actor = ActorNetworkX(perception_hidden_dim, self.action_dim)
        self.critic = CriticNetworkX(perception_hidden_dim, self.action_dim)

    def forward_actor(self, obs):
        return self.actor(obs)

    def forward_critic(self, obs, action):
        return self.critic(obs, action)

    def forward_perception(self, obs_in):
        return self.perception(self.scale_image(obs_in))

    def forward(self, obs_in):
        obs = self.forward_perception(obs_in)
        action = self.forward_actor(obs)
        value = self.forward_critic(obs, action)
        return (action, value)

    def scale_image(self, obs):
        '''
        Given uint8 input from the environment, scale to float32 and
        divide by 256 to scale inputs between 0.0 and 1.0
        '''
        obs_visual, obs_flat = obs
        if obs_visual is None:
            return obs
        return (obs_visual / 256.0, obs_flat)
        