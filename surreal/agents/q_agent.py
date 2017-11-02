"""
Actor function
"""
import torch
import random
from torch.autograd import Variable
from surreal.utils.torch_util import to_scalar
import threading


class QAgent(object):
    # TODO: formalize lock attr in BaseAgent
    def __init__(self, q_func, action_dim, init_eps=0):
        self.q_func = q_func
        self._eval = 'train'
        self.action_dim = action_dim
        self.eps = init_eps
        assert 0 <= self.eps <= 1
        # TODO standardize
        self._lock = threading.Lock()

    def set_train(self):
        self._eval = 'train'

    def get_lock(self):
        return self._lock

    def set_eval(self, stochastic, *, eps=0):
        """
        If determinstic, ignore `eps`
        """
        self._eval = 'eval-' + ('s' if stochastic else 'd')
        if self._eval == 'eval-s':
            self.eps = eps

    def act(self, obs, *, vectorize=False):
        if self._eval == 'eval-d' or random.random() > self.eps:
            assert torch.is_tensor(obs)
            if vectorize:
                # add a fake batch dim to new_obs
                # same effect: new_obs = new_obs.unsqueeze(0)
                obs = obs[None]
            obs = Variable(obs, volatile=True)
            with self._lock:
                q_values = self.q_func(obs)
            return to_scalar(q_values.data.max(1)[1].cpu())
        else: # random exploration
            return random.randrange(self.action_dim)

    def save(self, fname):
        with self._lock:
            self.q_func.save(fname)

    @classmethod
    def load(cls, q_module_class, fname, *args, **kwargs):
        """
        q_module_class: instance of surreal.utils.torch_util.Module
        """
        q_func = q_module_class.load(fname)
        return cls(q_func, *args, **kwargs)
