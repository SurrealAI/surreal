import numpy as np
import time
import threading
from surreal.agents import Agent


class DummyAgent(Agent):
    """
    Base agent class
    """
    def __init__(self, rank):
        """Each agent has a body and a head"""
        super().__init__('dummy', rank)
        self.dummy_matrix = np.ones([4]).astype(np.float32) * self._rank * 100

    def get_lock(self):
        return threading.Lock()

    def forward(self, obs):
        """Predict the next action by the policy

        Args:
            obs: current observation
        Returns:
            action output from the policy
        """
        time.sleep(0.1)
        return int(obs[0,0,0] - self.dummy_matrix[0,0,0]) % 10

    def terminate(self):
        """Clean up agent at the end of episode"""
        pass
