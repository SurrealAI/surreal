"""
Sample pointers from replay buffer and pull the actual observations
"""
from surreal.comm import RedisClient, ObsPack, ExpPack, to_str


class ExpDownloader:
    def __init__(self, redis_client):
        self.client = redis_client

    def download(self, exp_pointer, obs_pointers):
        assert isinstance(obs_pointers, list)
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