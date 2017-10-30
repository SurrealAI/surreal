"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
from .listener import PSListener
import time
import surreal.utils as U
import threading


class TorchPSListener(PSListener):
    def __init__(self, redis_client, ps_name, net, lock):
        super().__init__(redis_client, ps_name)
        U.assert_type(net, U.Module)
        U.assert_type(lock, type(threading.Lock()))
        self.net = net
        self._lock = lock

    def update(self, binary, message):
        with self._lock:
            self.net.parameters_from_binary(binary)

