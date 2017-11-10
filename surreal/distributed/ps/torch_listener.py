"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
import time
import surreal.utils as U
from .listener import Listener
from .module_dict import ModuleDict


class TorchListener(Listener):
    def __init__(self,
                 redis_client,
                 module_dict,
                 name='ps',
                 *, debug=False):
        super().__init__(redis_client=redis_client, name=name)
        self._module_dict = ModuleDict(module_dict)
        self._debug = debug

    def update(self, binary, message=''):
        """
        Override method.
        """
        self._module_dict.loads(binary)
        if self._debug:
            print('RECEIVED', message,
                  'BINARY_HASH', U.binary_hash(binary))
