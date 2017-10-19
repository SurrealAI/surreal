"""
A template class that defines base queue APIs
"""


class Queue(object):
    """
    Base queue class
    """

    def __init__(self, name="queue"):
        self._name = name
        self._type = None

    def add(self, transitions):
        """Add a new experience into queue

        Args:
            transitions: a list of (s, a, s', r) encoded by protobuf
        """
        raise NotImplementedError()

    def empty(self):
        """Empty the queue and reset"""
        raise NotImplementedError()

    def sample(self, batch_size):
        """Sample a mini-batch of experiences from the queue
        
        Args:
            batch_size: number of experiences to be sampled
        """
        raise NotImplementedError()