"""
A template class that defines base agent APIs
"""


class Agent(object):
    """
    Base agent class
    """

    def __init__(self, name="agent", rank=0):
        """Each agent has a body and a head"""
        self._name = name
        self._rank = rank
        self._body = None
        self._head = None

    def initialize(self):
        """Initialize agent at the beginning of episode"""
        raise NotImplementedError()

    def forward(self, obs):
        """Predict the next action by the policy

        Args:
            obs: current observation
        Returns:
            action output from the policy
        """
        raise NotImplementedError()

    def terminate(self):
        """Clean up agent at the end of episode"""
        raise NotImplementedError()
