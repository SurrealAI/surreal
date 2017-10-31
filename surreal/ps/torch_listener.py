"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
from .listener import Listener
import time
import surreal.utils as U
import threading


class TorchListener(Listener):
    def __init__(self,
                 redis_client,
                 net,
                 lock,
                 name='ps',
                 *, debug=False):
        super().__init__(redis_client, name)
        U.assert_type(net, U.Module)
        U.assert_type(lock, type(threading.Lock()))
        self.net = net
        self._lock = lock
        self._debug = debug

    def update(self, binary, message):
        with self._lock:
            self.net.parameters_from_binary(binary)
            if self._debug:
                print('RECEIVED', message,
                      'BINARY_HASH', U.binary_hash(binary))

