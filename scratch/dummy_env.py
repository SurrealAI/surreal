"""
A template class that defines base environment APIs
"""
import numpy as np
from surreal.envs.base import Env
import time


class DummyEnv(Env):
    """
    Base environment class
    """
    def __init__(self, base_matrix, sleep=0.5):
        self.matrix = base_matrix
        self.sleep = sleep
        super().__init__()

    def initialize(self):
        """Initialize environment at the beginning of episode"""
        pass

    def step(self, action):
        """Make a step in the environment

        Args:
            action: the next action to take
        """
        self.matrix = self.matrix + np.ones([4]).astype(np.float32)
        time.sleep(self.sleep)
        return self.matrix, action*0.1, bool(action%3==0), {'yoyo': action*0.3}

    def terminate(self):
        """Clean up when an episode ends"""
        pass
