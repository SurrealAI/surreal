import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np
import torchx.nn as nnx

from .model_builders import *
from .z_filter import ZFilter

import itertools

class DiagGauss(object):
    '''
        Class that encapsulates Diagonal Gaussian Probability distribution
        Attributes:
            d: action dimension
        Member Functions:
            loglikelihood
            likelihood
            kl
            entropy
            sample
            maxprob
    '''
    def __init__(self, action_dim):
        self.d = action_dim

    def loglikelihood(self, a, prob):
        '''
            Method computes loglikelihood of action (a) given probability (prob)
        '''
        if len(a.size()) == 3:
            a = a.view(-1, self.d)
            prob = prob.view(-1, 2 * self.d)

        mean0 = prob[:, :self.d]
        std0 = prob[:, self.d:]
        return - 0.5 * (((a - mean0) / std0).pow(2)).sum(dim=1, keepdim=True) - 0.5 * np.log(
            2.0 * np.pi) * self.d - std0.log().sum(dim=1, keepdim=True)

    def likelihood(self, a, prob):
        '''
            Method computes likelihood of action (a) given probability (prob)
        '''
        return torch.clamp(self.loglikelihood(a, prob).exp(), min=1e-5)

    def kl(self, prob0, prob1):
        '''
            Method computes KL Divergence of between two probability distributions
            Note: this is D_KL(prob0 || prob1), not D_KL(prob1 || prob0)
        '''
        if len(prob0.size()) == 3:
            prob0 = prob0.view(-1, 2 * self.d)
            prob1 = prob1.view(-1, 2 * self.d)

        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return ((std1 / std0).log()).sum(dim=1) + (
            (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2))).sum(dim=1) - 0.5 * self.d

    def entropy(self, prob):
        '''
            Method computes entropy of a given probability (prob)
        '''
        if len(prob.size()) == 3:
            prob = prob.view(-1, 2 * self.d)

        std_nd = prob[:, self.d:]
        return 0.5 * std_nd.log().sum(dim=1) + .5 * np.log(2 * np.pi * np.e) * self.d

    def sample(self, prob):
        '''
            Method samples actions from probability distribution
        '''
        if len(prob.shape) == 3:
            prob_shape = prob.shape
            prob = prob.reshape(-1, self.d * 2)
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d) * std_nd + mean_nd

    def maxprob(self, prob):
        '''
            Method deterministically sample actions of maximum likelihood
        '''
        if len(prob.shape) == 3:
            return prob[:, :, self.d]
        return prob[:, :self.d]


class PPOModel(nnx.Module):
    '''
        PPO Model class that wraps aroud the actor and critic networks
        Attributes:
            actor: Actor network, see surreal.model.model_builders.builders
            critic: Critic network, see surreal.model.model_builders.builders
            z_filter: observation z_filter. see surreal.model.z_filter
        Member functions:
            update_target_param: updates kept parameters to that of another model
            update_target_param: updates kept z_filter to that of another model
            forward_actor: forward pass actor to generate policy with option
                to use z-filter
            forward_actor: forward pass critic to generate policy with option
                to use z-filter
            z_update: updates Z_filter running obs mean and variance
    '''
    def __init__(self,
                 obs_spec,
                 action_dim,
                 model_config,
                 use_cuda,
                 init_log_sig=0,
                 use_z_filter=False,
                 if_pixel_input=False,
                 rnn_config=None):
        super(PPOModel, self).__init__()

        # hyperparameters
        self.obs_spec = obs_spec
        self.action_dim = action_dim
        self.model_config = model_config
        self.use_z_filter = use_z_filter
        self.init_log_sig = init_log_sig
        self.if_pixel_input = if_pixel_input
        self.rnn_config = rnn_config

        # compute low dimensional feature dimension
        self.low_dim = 0
        if 'low_dim' in self.obs_spec.keys():
            for key in self.obs_spec['low_dim'].keys():
                self.low_dim += self.obs_spec['low_dim'][key][0]

        # optional CNN stem feature extractor
        self.cnn_stem = None
        if self.if_pixel_input:
            self.cnn_stem = CNNStemNetwork(self.obs_spec['pixel']['camera0'],
                                           self.model_config.cnn_feature_dim)

        # optional LSTM stem feature extractor
        self.rnn_stem = None
        if self.rnn_config.if_rnn_policy:
            rnn_insize = self.low_dim + (self.model_config.cnn_feature_dim if self.if_pixel_input else 0)
            self.rnn_stem = nn.LSTM(rnn_insize,
                                    self.rnn_config.rnn_hidden,
                                    self.rnn_config.rnn_layer,
                                    batch_first=True)
            if use_cuda:
                device = torch.device("cuda")
                self.rnn_stem = self.rnn_stem.to(device)

        # computing final feature dimension for leaf actor/critic network
        input_size = self.low_dim + (self.model_config.cnn_feature_dim if self.if_pixel_input else 0)
        input_size = self.rnn_config.rnn_hidden if self.rnn_config.if_rnn_policy else input_size

        self.actor = PPO_ActorNetwork(input_size, 
                                      self.action_dim, 
                                      self.model_config.actor_fc_hidden_sizes,
                                      self.init_log_sig)
        self.critic = PPO_CriticNetwork(input_size,
                                      self.model_config.critic_fc_hidden_sizes)
        if self.use_z_filter:
            assert self.low_dim > 0, "No low dimensional input, please turn off z-filter"
            self.z_filter = ZFilter(self.obs_spec)

    def _gather_low_dim_input(self, obs):
        '''
            Concatenate (along 2nd dimension) all the low-dimensional
            (propioceptive) features from input observation tuple
            Args:
                obs: dictionary of tensors
        '''
        if 'low_dim' not in obs.keys(): return None
        list_obs_ld = [obs['low_dim'][key] for key in obs['low_dim'].keys()]
        obs_low_dim = torch.cat(list_obs_ld, -1)
        return obs_low_dim

    def clear_actor_grad(self):
        '''
            Method that clears all gradients from all the parameters from
            actor, cnn_stem (optional), and rnn_stem (optional)
        '''
        self.actor.zero_grad()
        if self.if_pixel_input:
            self.cnn_stem.zero_grad()
        if self.rnn_config.if_rnn_policy:
            self.rnn_stem.zero_grad()

    def clear_critic_grad(self):
        '''
            Method that clears all gradients from all the parameters from
            critic, cnn_stem (optional), and rnn_stem (optional)
        '''
        self.critic.zero_grad()
        if self.if_pixel_input:
            self.cnn_stem.zero_grad()
        if self.rnn_config.if_rnn_policy:
            self.rnn_stem.zero_grad()

    def get_actor_params(self):
        '''
            Method that returns generator that contains all the parameters from
            actor, cnn_stem (optional), and rnn_stem (optional)
        '''
        params = self.actor.parameters()
        if self.if_pixel_input:
            params = itertools.chain(params, self.cnn_stem.parameters())
        if self.rnn_config.if_rnn_policy:
            params = itertools.chain(params, self.rnn_stem.parameters())
        return params

    def get_critic_params(self):
        '''
            Method that returns generator that contains all the parameters from
            critic, cnn_stem (optional), and rnn_stem (optional)
        '''
        params = self.critic.parameters()
        if self.if_pixel_input:
            params = itertools.chain(params, self.cnn_stem.parameters())
        if self.rnn_config.if_rnn_policy:
            params = itertools.chain(params, self.rnn_stem.parameters())
        return params

    def update_target_params(self, net):
        '''
            updates kept parameters to that of another model
            Args:
                net: another PPO_Model instance
        '''
        self.actor.load_state_dict(net.actor.state_dict())
        self.critic.load_state_dict(net.critic.state_dict())

        if self.rnn_config.if_rnn_policy:
            self.rnn_stem.load_state_dict(net.rnn_stem.state_dict())

        if self.if_pixel_input:
            self.cnn_stem.load_state_dict(net.cnn_stem.state_dict())

        if self.use_z_filter:
            self.z_filter.load_state_dict(net.z_filter.state_dict())

    def update_target_z_filter(self, net):
        '''
            updates kept z-filter to that of another model
            Args:
                net: another PPO_Model instance
        '''
        if self.use_z_filter:
            self.z_filter.load_state_dict(net.z_filter.state_dict())

    def forward_actor(self, obs, cells=None):
        '''
            forward pass actor to generate policy with option to use z-filter
            Args:
                obs: dictionary of batched observations
                cells: tuple of hidden and cell state for LSTM stem. optional
            Returns:
                The output of actor network
        '''
        obs_list = []
        obs_flat = self._gather_low_dim_input(obs)
        if self.use_z_filter:
            obs_flat = self.z_filter.forward(obs_flat)
        obs_list.append(obs_flat)

        if self.if_pixel_input:
            # right now assumes only one camera angle.
            obs_pixel = obs['pixel']['camera0']
            obs_pixel = self._scale_image(obs_pixel)
            obs_pixel = self.cnn_stem(obs_pixel)
            obs_list.append(obs_pixel)

        obs = torch.cat([ob for ob in obs_list if ob is not None], dim=-1)
            
        if self.rnn_config.if_rnn_policy:
            obs, _ = self.rnn_stem(obs, cells)
            obs = obs.contiguous()

        action = self.actor(obs)
        return action

    def forward_critic(self, obs, cells=None):
        '''
            forward pass critic to generate policy with option to use z-filter
            Note: assumes input has either shape length 2 or 4 without RNN and
            length 3 or 5 with RNN depending on if image based training is used
            Args: 
                obs: dictionary of batched observations
                cells: tuple of hidden and cell state for LSTM stem. optional
            Returns:
                output of critic network
        '''
        obs_list = []
        obs_flat = self._gather_low_dim_input(obs)
        if self.use_z_filter:
            obs_flat = self.z_filter.forward(obs_flat)
        obs_list.append(obs_flat)

        if self.if_pixel_input:
            # right now assumes only one camera angle.
            obs_pixel = obs['pixel']['camera0']
            obs_pixel = self._scale_image(obs_pixel)
            obs_pixel = self.cnn_stem(obs_pixel)
            obs_list.append(obs_pixel)

        obs = torch.cat([ob for ob in obs_list if ob is not None], dim=-1)

        if self.rnn_config.if_rnn_policy:
            obs, _ = self.rnn_stem(obs, cells)
            obs = obs.contiguous()

        value = self.critic(obs)
        return value

    def forward_actor_expose_cells(self, obs, cells=None):
        '''
            forward pass critic to generate policy with option to use z-filter
            also returns an updated LSTM hidden/cell state when necessary
            Args: 
                obs: dictionary of batched observations
                cells: tuple of hidden and cell state for LSTM stem. optional
            Returns:
                output of critic network
        '''
        obs_list = []
        obs_flat = self._gather_low_dim_input(obs)
        if self.use_z_filter:
            obs_flat = self.z_filter.forward(obs_flat)
        obs_list.append(obs_flat)

        if self.if_pixel_input:
            # right now assumes only one camera angle.
            obs_pixel = obs['pixel']['camera0']
            obs_pixel = self._scale_image(obs_pixel)
            obs_pixel = self.cnn_stem(obs_pixel) 
            obs_list.append(obs_pixel)

        obs = torch.cat([ob for ob in obs_list if ob is not None], dim=-1)

        if self.rnn_config.if_rnn_policy:
            obs = obs.view(1, 1, -1) # assume input is shape (1, obs_dim)
            obs, cells = self.rnn_stem(obs, cells)
            
            # .detach() is necessary here to prevent overflow of memory
            # otherwise rollout in length of thousands will prevent previously
            # accumulated hidden/cell states from being freed.
            cells = (cells[0].detach(),cells[1].detach())
            obs = obs.contiguous()  
            obs = obs.view(-1, self.rnn_config.rnn_hidden)

        action = self.actor(obs) # shape (1, action_dim)
        return action, cells

    def z_update(self, obs):
        '''
            updates Z_filter running obs mean and variance
            Args:
                obs: dictionary of batched observations
        '''
        if self.use_z_filter:
            obs_flat = self._gather_low_dim_input(obs)
            self.z_filter.z_update(obs_flat)
        else:
            raise ValueError('Z_update called when network is set to not use z_filter')

    def _scale_image(self, obs):
        '''
            Given uint8 input from the environment, scale to float32 and
            divide by 255 to scale inputs between 0.0 and 1.0
            Args:
                obs: dictionary of batched observations
        '''
        return obs / 255.0
