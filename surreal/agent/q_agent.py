"""
Actor function
"""
import torch
import random
from torch.autograd import Variable
from surreal.utils.pytorch import to_scalar
import threading
from .base import Agent


class QAgent(Agent):
    def __init__(self, model, action_mode, action_dim):
        super().__init__(model, action_mode)
        self.action_dim = action_dim

    def _act(self, obs, model, action_mode, eps=0.):
        assert torch.is_tensor(obs)
        if action_mode == 'eval-d' or random.random() > eps:
            obs = obs[None]  # vectorize
            obs = Variable(obs, volatile=True)
            q_values = model(obs)
            return to_scalar(q_values.data.max(1)[1].cpu())
        else:  # random exploration
            return random.randrange(self.action_dim)
