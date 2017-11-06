"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
from .listener import Listener
import time
from surreal.agent.base import Agent
import surreal.utils as U
import threading


class TorchListener(Listener):
    def __init__(self,
                 redis_client,
                 agent,
                 name='ps',
                 *, debug=False):
        super().__init__(redis_client, name)
        U.assert_type(agent, Agent)
        self._model = agent.get_model()
        self._lock = agent.get_lock()
        self._debug = debug

    def update(self, binary, message=''):
        """
        Override method.
        """
        with self._lock:
            self._model.parameters_from_binary(binary)
            if self._debug:
                print('RECEIVED', message,
                      'BINARY_HASH', U.binary_hash(binary))

