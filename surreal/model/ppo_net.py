import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np

from .model_builders import *
from .z_filter import ZFilter

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


class PPOModel(U.Module):
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
                 init_log_sig,
                 obs_config,
                 action_dim,
                 use_z_filter,
                 pixel_config,
                 rnn_config,
                 use_cuda):
        super(PPOModel, self).__init__()

        # hyperparameters
        self.obs_spec, self.input_config = obs_config
        self.action_dim = action_dim
        self.use_z_filter = use_z_filter
        self.init_log_sig = init_log_sig
        self.pixel_config = pixel_config
        self.rnn_config = rnn_config

        self.low_dim = U.observation.get_low_dim_shape(self.obs_spec, self.input_config)
        self.cnn_stem = None
        self.if_pixel_input = self.pixel_config is not None
        if self.if_pixel_input:
            self.cnn_stem = PerceptionNetwork(self.obs_spec['pixels'],
                                              self.pixel_config.perception_hidden_dim,
                                              use_layernorm=self.pixel_config.use_layernorm)
            if use_cuda:
                self.cnn_stem = self.cnn_stem.cuda()

        self.rnn_stem = None
        if self.rnn_config.if_rnn_policy:
            self.rnn_stem = nn.LSTM(self.low_dim if not self.if_pixel_input else \
                                    self.pixel_config.perception_hidden_dim,
                                    self.rnn_config.rnn_hidden,
                                    self.rnn_config.rnn_layer,
                                    batch_first=True)
            if use_cuda:
                self.rnn_stem = self.rnn_stem.cuda()

        input_size = self.pixel_config.perception_hidden_dim if self.if_pixel_input else self.low_dim
        input_size = self.rnn_config.rnn_hidden if self.rnn_config.if_rnn_policy else input_size

        self.actor = PPO_ActorNetwork(input_size, 
                                      self.action_dim, 
                                      self.init_log_sig, 
                                      self.cnn_stem,
                                      self.rnn_stem)
        self.critic = PPO_CriticNetwork(input_size, self.cnn_stem, self.rnn_stem)
        if self.use_z_filter:
            assert self.low_dim > 0, "No low dimensional input, please turn off z-filter"
            self.z_filter = ZFilter(self.obs_spec, 
                                    self.input_config, 
                                    pixel_input=self.if_pixel_input,
                                    use_cuda=use_cuda)

    def update_target_params(self, net):
        '''
            updates kept parameters to that of another model
            Args:
                net: another PPO_Model instance
        '''
        self.actor.load_state_dict(net.actor.state_dict())
        self.critic.load_state_dict(net.critic.state_dict())
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
                obs -- batch of observations
            Returns:
                The output of actor network
        '''
        obs_list = []
        obs_flat = U.observation.gather_low_dim_input(obs, self.input_config)
        if self.use_z_filter:
            obs_flat = self.z_filter.forward(obs_flat)
        obs_list.append(obs_flat)

        if self.if_pixel_input:
            # right now assumes only one camera angle.
            obs_pixel = obs[self.input_config['pixel'][0]]
            if self.pixel_config.if_uint8:
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
                obs -- batch of observations
            Returns:
                output of critic network
        '''
        obs_list = []
        obs_flat = U.observation.gather_low_dim_input(obs, self.input_config)
        if self.use_z_filter:
            obs_flat = self.z_filter.forward(obs_flat)
        obs_list.append(obs_flat)

        if self.if_pixel_input:
            # right now assumes only one camera angle.
            obs_pixel = obs[self.input_config['pixel'][0]]
            if self.pixel_config.if_uint8:
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
                obs -- batch of observations
            Returns:
                output of critic network
        '''
        obs_list = []
        obs_flat = U.observation.gather_low_dim_input(obs, self.input_config)
        if self.use_z_filter:
            obs_flat = self.z_filter.forward(obs_flat)
        obs_list.append(obs_flat)

        if self.if_pixel_input:
            # right now assumes only one camera angle.
            obs_pixel = obs[self.input_config['pixel'][0]]
            if self.pixel_config.if_uint8:
                obs_pixel = self._scale_image(obs_pixel)
            obs_pixel = self.cnn_stem(obs_pixel) 
            obs_list.append(obs_pixel)

        obs = torch.cat([ob for ob in obs_list if ob is not None], dim=-1)

        if self.rnn_config.if_rnn_policy:
            obs = obs.view(1, 1, -1) # assume input is shape (1, obs_dim)
            obs, cells = self.rnn_stem(obs, cells)
            
            # Note that this is effectively the same of a .detach() call.
            # .detach() is necessary here to prevent overflow of memory
            # otherwise rollout in length of thousands will prevent previously
            # accumulated hidden/cell states from being freed.
            cells = (Variable(cells[0].data),Variable(cells[1].data))
            obs = obs.contiguous()  
            obs = obs.view(-1, self.rnn_config.rnn_hidden)

        action = self.actor(obs) # shape (1, action_dim)
        return action, cells

    def z_update(self, obs):
        '''
            updates Z_filter running obs mean and variance
            Args: obs -- batch of observations
        '''
        if self.use_z_filter:
            obs_flat = U.observation.gather_low_dim_input(obs, self.input_config)
            self.z_filter.z_update(obs_flat)
        else:
            raise ValueError('Z_update called when network is set to not use z_filter')

    def _scale_image(self, obs):
        '''
        Given uint8 input from the environment, scale to float32 and
        divide by 256 to scale inputs between 0.0 and 1.0
        '''
        return obs / 256.0