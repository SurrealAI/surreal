"""
Learner: pushes parameters to key "ps" and
    param info to hashmap key "psinfo" on Redis.
Agent: pulls parameters from key "ps"
Evaluator: pulls param info from "psinfo" and do diagnostics.
"""
import time
import pickle
import surreal.utils as U
from .module_dict import ModuleDict


_LUA_FETCH_WHEN_CHANGED = """
local psname, pshash
psname = KEYS[1]
pshash = redis.call('GET', KEYS[2])
if pshash == ARGV[1] then
-- do not download from PS if the signature has not changed (i.e. PS not updated)    
    return {0, pshash}
else
    return {redis.call('GET', psname), pshash}
end
"""


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
        if not isinstance(module_dict, ModuleDict):
            module_dict = ModuleDict(module_dict)
        self._module_dict = module_dict
        self._name = name
        self._info_name = name + '_info'
        self._hash_name = name + '__hash__'  # for Lua script
        self._last_hash = ''  # do not fetch until hash changed
        self._fetch_when_changed = redis_client.register_script(
            _LUA_FETCH_WHEN_CHANGED
        )

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
            self._info_name: pickle.dumps(info),
            self._hash_name: info['hash']
        })

    def pull(self):
        """
        Called by agent. Pulls from PS ONLY WHEN the parameter hash changes to
        prevent duplicate fetching. No-op when duplicate.

        Returns:
            bool: True if the parameter is fetched, False if duplicate.
        """
        binary, self._last_hash = self._fetch_when_changed(
            keys=[self._name, self._hash_name],
            args=[self._last_hash]  # fetch only when hash != remote_hash
        )
        if binary in [0, None]:  # parameter update not yet available
            return False
        else:
            self._module_dict.loads(binary)
            return True

    def pull_info(self):
        info = self._client.get(self._info_name)
        if info is None:
            return None  # parameter info not yet available
        else:
            return pickle.loads(info)
