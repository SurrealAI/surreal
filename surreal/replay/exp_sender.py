"""
Sample pointers from replay buffer and pull the actual observations
"""
from surreal.comm import RedisClient, PointerPack, ObsPack, ExpPack, to_str


class ExpSender:
    def __init__(self, redis_client, queue_name):
        assert isinstance(redis_client, RedisClient)
        self.client = redis_client
        self.queue_name = queue_name

    def send(self, obses, action, reward, done, info, replay_info):
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
            obs_pointer, binary = ObsPack(obs).serialize()
            redis_mset[obs_pointer] = binary
            obs_pointers.append(obs_pointer)
        # experience pack
        exp = ExpPack(obs_pointers, action, reward, done, info)
        exp_pointer, binary = exp.serialize()
        redis_mset[exp_pointer] = binary
        self.client.mset(redis_mset)
        # pointer pack
        ppack = PointerPack(obs_pointers, exp_pointer, replay_info)
        _, binary = ppack.serialize()
        self.client.push_to_queue(self.queue_name, binary)
