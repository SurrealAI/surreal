"""
Learner: pushes parameters to key "ps" and
    param info to hashmap key "psinfo" on Redis.
Agent: pulls parameters from key "ps"
Evaluator: pulls param info from "psinfo" and do diagnostics.
"""
import time
import pickle
from surreal.distributed import RedisClient
import surreal.utils as U
from .module_dict import ModuleDict


class ParameterServer(object):
    def __init__(self,
                 redis_client,
                 module_dict,
                 name='ps'):
        """
        Args:
            name: key that points to the parameter binary on Redis.
                "<name>info" will be the key to the info Redis hashmap.
                e.g. "psinfo" -> {'time': 32541.6, 'iteration': 1200}
        """
        U.assert_type(redis_client, RedisClient)
        self._client = redis_client
        self._module_dict = ModuleDict(module_dict)
        self._name = name
        self._info_name = name + 'info'

    def push(self, iteration, message=''):
        """
        Called by learner.

        Args:
            iteration: current learning iteration
            message: any pickleable data
        """
        binary = self._module_dict.dumps()
        info = {
            'time': time.time(),
            'iteration': iteration,
            'message': message,
            'hash': U.binary_hash(binary)
        }
        self._client.mset({
            self._name: binary,
            self._info_name: pickle.dumps(info)
        })

    def pull(self):
        """
        Called by agent.
        """
        binary = self._client.get(self._name)
        if binary is not None:  # parameter update not yet available
            self._module_dict.loads(binary)

    def pull_info(self):
        info = self._client.get(self._info_name)
        if info is None:
            return None  # parameter info not yet available
        else:
            return pickle.loads(info)
