"""
Actor function
"""
import torch
import random
from torch.autograd import Variable
from surreal.utils.pytorch import to_scalar
import threading
from .base import Agent
import numpy as np


class DDPGAgent(Agent):

    def __init__(self, model, agent_mode, action_dim):
        super().__init__(model, agent_mode)
        self.action_dim = action_dim

        # Ornstein-Uhlenbeck noise for exploration
        self.use_ou_noise = False
        self.noise = torch.zeros(1, self.action_dim)

        self.logsig = -1.0

    def _act(self, obs, model, action_mode, eps=0.):

        obs = Variable(obs.unsqueeze(0))
        action = self.model.actor(obs)

        if action_mode != 'noiseless':
            std = float(np.exp(self.logsig))
            noise_random = torch.zeros(1, self.action_dim).normal_(std=std)
            if self.use_ou_noise:
                self.noise = self.ou_noise + noise_random
            else:
                self.noise = noise_random
            action.data.add_(self.noise).clamp_(-1, 1)

        return action.data.numpy().squeeze()
