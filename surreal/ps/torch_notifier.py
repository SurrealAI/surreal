from .notifier import PSNotifier
from surreal.comm import RedisClient
import surreal.utils as U


class TorchPSNotifier(PSNotifier):
    def __init__(self, redis_client, ps_name, net):
        super().__init__(redis_client, ps_name)
        U.assert_type(net, U.Module)
        self.net = net

    def get_serialized_state(self):
        return self.net.parameters_to_binary()

