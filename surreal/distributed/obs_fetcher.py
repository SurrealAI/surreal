"""
Sample pointers from replay buffer and pull the actual observations
"""
from .packs import ObsPack, RedisMissingValue


class ObsFetcher(object):
    def __init__(self, redis_client):
        self._client = redis_client

    def fetch(self, exp_dicts):
        """
        Args:
            exp_dicts: list of exp_dicts with 'obs_pointers' field
            NOTE if exp_dicts already have 'obs' field, the call is no-op
        Returns:
            fill out each 'obs' field, delete 'obs_pointers' key
        """
        assert isinstance(exp_dicts, list)
        assert len(exp_dicts) > 0, 'exp_dicts download list cannot be empty'
        assert all(isinstance(exp, dict) for exp in exp_dicts)
        # prevent reinserting the actual heavy new_obs back into exps
        exp_dicts = [exp.copy() for exp in exp_dicts]
        all_obs_pointers = []
        for exp in exp_dicts:
            if 'obs' in exp:
                # obs already present, no need to re-fetch
                continue
            else:
                assert 'obs_pointers' in exp
                all_obs_pointers.extend(exp['obs_pointers'])
        all_obs = self._client.mget(all_obs_pointers)
        try:
            all_obs = [ObsPack.deserialize(obs) for obs in all_obs]
        except RedisMissingValue:
            # print('ERROR fetch obs_pointers', all_obs_pointers)
            raise RedisMissingValue
        for exp in exp_dicts:
            if 'obs' in exp:
                continue
            num_obs = len(exp['obs_pointers'])
            exp['obs'] = all_obs[:num_obs]
            del all_obs[:num_obs]
        assert len(all_obs) == 0, 'should be empty after all exp'
        return exp_dicts

