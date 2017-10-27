"""
Sample pointers from replay buffer and pull the actual observations
"""
from surreal.comm import RedisClient, ObsPack, ExpPack


class ExpSender:
    def __init__(self, redis_client, queue_name):
        assert isinstance(redis_client, RedisClient)
        self.client = redis_client
        self.queue_name = queue_name
        self.visited_obs = set() # avoid resending obs

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
            obs_pointer, binary = pack.get_key(), pack.serialize()
            if obs_pointer not in self.visited_obs:
                self.visited_obs.add(obs_pointer)
                redis_mset[obs_pointer] = binary
            obs_pointers.append(obs_pointer)
        # experience pack
        pack = ExpPack(obs_pointers, action, reward, done, info)
        exp_pointer, binary = pack.get_key(), pack.serialize()
        redis_mset[exp_pointer] = binary
        self.client.mset(redis_mset)
        # send to queue
        self.client.push_to_queue(self.queue_name, binary)

