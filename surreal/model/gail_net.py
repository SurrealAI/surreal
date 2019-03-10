import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np
import torchx
import torchx.nn as nnx

from .model_builders import *
from .z_filter import ZFilter
from .ppo_net import DiagGauss, PPOModel

import itertools


class GAILModel(nnx.Module):
    """
        The GAIL model just defines a discriminator.
    """
    def __init__(self,
                 obs_spec,
                 action_dim,
                 model_config,
                 use_cuda,
                 use_z_filter=False):

        super(GAILModel, self).__init__()
        self.obs_spec = obs_spec
        self.action_dim = action_dim
        self.model_config = model_config
        self.use_z_filter = use_z_filter

        # add a discriminator network
        self.obj_dim = self.obs_spec["low_dim"]["flat_inputs"][0]
        assert self.obj_dim > 0, "Need to have low-dim features for GAIL, add 'low-dim' \
                                  to the observation dictionary"
        print(self.obj_dim, self.model_config.discriminator_hidden_sizes)
        self.discriminator = GAIL_Discriminator(self.obj_dim + self.action_dim,
                                                self.model_config.discriminator_hidden_sizes)

    #def _gather_object_features(self, obs):
    #    """
    #        Concatenate (along 2nd dimension) all the low-dimensional
    #        object-centric features from input observation dict
    #    """
    #    if 'object' not in obs.keys(): return None
    #    list_obs_obj = [obs['object'][key] for key in obs['object'].keys()]
    #    obs_obj = torch.cat(list_obs_obj, -1)
    #    return obs_obj

    def clear_discriminator_grad(self):
        '''
            Method that clears all gradients from all the parameters from discriminator.
        '''
        self.discriminator.zero_grad()

    def forward_discriminator(self, obs):
        '''
            forward pass discriminator to classify states as expert or policy (rewards for RL)
            Note: assumes input has either shape length 2 or 4 
                  depending on if image based training is used
            Args: 
                obs -- batch of observations
            Returns:
                output of discriminator network
        '''

        ### TODO: should we have a z-filter on the discriminator? ###
        #obs = self._gather_object_features(obs)
        # if self.use_z_filter:
        #     obs = self.z_filter.forward(obs)
        logits = self.discriminator(obs)
        return logits

    def get_discriminator_params(self):
        """
            Method that returns generator that contains all the parameters from
            discriminator
        """
        return self.discriminator.parameters()

    def get_discriminator_reward(self, obs):
        """
            Get the value of the reward from the discriminator
        """
        logits = self.forward_discriminator(obs)
        reward = 1. - F.sigmoid(logits) + 1e-8
        reward = - reward.log()
        return reward


