import surreal.utils as U
from .broadcaster import Broadcaster
from .module_dict import ModuleDict


class TorchBroadcaster(Broadcaster):
    def __init__(self,
                 redis_client,
                 module_dict,
                 name='ps',
                 *, debug=False):
        super().__init__(redis_client=redis_client, name=name)
        self._module_dict = ModuleDict(module_dict)
        self._debug = debug

    def broadcast(self, message=''):
        binary = self._module_dict.dumps()
        if self._debug:
            print('BROADCAST', message, 'BINARY_HASH', U.binary_hash(binary))
        super().broadcast(binary, message)
