"""
A template class that defines base trainer APIs
"""


class Trainer(object):
    """
    Base trainer class
    """

    def __init__(self, name="trainer"):
        self._name = name
        self._loss = None
        self._agent = None
        self._optimizer = None
        self._checkpoint_manager = None
