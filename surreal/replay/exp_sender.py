"""
Sample pointers from replay buffer and pull the actual observations
"""
from surreal.comm import RedisClient, ObsPack, ExpPack, to_str


class ExpSender:
    def __init__(self, redis_client):
        self.client = redis_client

    def send(self, exp_dict):
        """
        Args:
            exp_dict: {obses: [np_image0, np_image1], action, reward, info}

        - Send the observations with their hash as key
        - Send the experience tuple with its hash as key
        - Send the pointer pack to Redis queue
        """
        assert isinstance(exp_dict, dict)



        exp, *obses = self.client.mget([exp_pointer] + obs_pointers)
        exp = ExpPack.deserialize(exp)
        assert len(obs_pointers) == len(exp.obs_pointers) # expected pointes
        actual_pointers = {to_str(p) for p in obs_pointers}
        downloaded_pointers = {to_str(p) for p in exp.obs_pointers}
        assert actual_pointers == downloaded_pointers
        obses = [ObsPack.deserialize(obs).obs for obs in obses]
        exp = exp.get_data()
        exp['obses'] = obses
        del exp['obs_pointers']
        return exp