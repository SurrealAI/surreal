"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
from .listener import PSListener
import queue
import itertools
from time import sleep
from surreal.comm import to_str


class TorchPSListener(PSListener):
    def __init__(self, redis_client, ps_name):
        self.client = redis_client
        self.ps_name = ps_name
        self._listener_thread = None

    def unflatten_tensors(self, flat, tensors):
        """View a flat buffer using the sizes of tensors"""
        outputs = []
        offset = 0
        for tensor in tensors:
            numel = tensor.numel()
            outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
            offset += numel
        return tuple(outputs)

