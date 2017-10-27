"""
A template class that defines base environment APIs
"""


class Env(object):
    """
    Base environment class
    """

    def __init__(self, name="env"):
        self._name = name

    def initialize(self):
        """Initialize environment at the beginning of episode"""
        raise NotImplementedError()

    def step(self, action):
        """Make a step in the environment

        Args:
            action: the next action to take
        """
        raise NotImplementedError()

    def terminate(self):
        """Clean up when an episode ends"""
        raise NotImplementedError()
