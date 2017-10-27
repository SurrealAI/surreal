"""
Sample pointers from replay buffer and pull the actual observations
"""
from surreal.comm import RedisClient, ObsPack, ExpPack, to_str


class ObsDownloader:
    def __init__(self, redis_client):
        self.client = redis_client

    def download(self, exp_dicts):
        """
        Args:
            exp_dicts: list of exp_dicts with 'obs_pointers' field
        Returns:
            fill out each 'obses' field, delete 'obs_pointers' key
        """
        assert isinstance(exp_dicts, list)
        assert len(exp_dicts) > 0, 'exp_dicts download list cannot be empty'
        assert all(isinstance(exp, dict) for exp in exp_dicts)
        # prevent reinserting the actual heavy obs back into exps
        exp_dicts = [exp.copy() for exp in exp_dicts]
        all_obs_pointers = []
        for exp in exp_dicts:
            all_obs_pointers.extend(exp['obs_pointers'])
        all_obs = self.client.mget(all_obs_pointers)
        all_obs = [ObsPack.deserialize(obs).data for obs in all_obs]
        for exp in exp_dicts:
            num_obs = len(exp['obs_pointers'])
            exp['obses'] = all_obs[:num_obs]
            del exp['obs_pointers']
            del all_obs[:num_obs]
        assert len(all_obs) == 0, 'should be empty after all exp'
        return exp_dicts

