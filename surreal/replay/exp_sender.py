"""
Sample pointers from replay buffer and pull the actual observations
"""
from surreal.distributed import RedisClient, ObsPack, ExpPointerPack, ExpFullPack


class ExpSender(object):
    def __init__(self, redis_client, queue_name, obs_cache_size=10000000):
        """
        obs_cache_size: max size of the cache of new_obs hashes so that we don't
            send duplicate new_obs to Redis.
        """
        assert isinstance(redis_client, RedisClient)
        self.client = redis_client
        self.queue_name = queue_name
        self._visited_obs = set() # avoid resending new_obs
        self._obs_cache_size = obs_cache_size

    def _add_to_visited(self, obs_pointer):
        self._visited_obs.add(obs_pointer)
        if len(self._visited_obs) > self._obs_cache_size:
            self._visited_obs.pop()

    def send(self, obses, action, reward, done, info):
        """
        Args:
            exp_dict: {obses: [np_image0, np_image1], action, reward, info}

        - Send the observations with their hash as key
        - Send the experience tuple with its hash as key
        - Send the PointerPack to Redis queue
        """
        redis_mset = {}
        # observation pack
        obs_pointers = []
        for obs in obses:
            pack = ObsPack(obs)
            obs_pointer, binary = pack.serialize()
            if obs_pointer not in self._visited_obs:
                self._add_to_visited(obs_pointer)
                redis_mset[obs_pointer] = binary
            obs_pointers.append(obs_pointer)
        # experience pack
        pack = ExpPointerPack(obs_pointers, action, reward, done, info)
        exp_pointer, binary = pack.serialize()
        redis_mset[exp_pointer] = binary
        self.client.mset(redis_mset)
        # send to queue
        self.client.push_to_queue(self.queue_name, binary)

