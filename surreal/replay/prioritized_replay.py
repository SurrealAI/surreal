import random
from .base import Replay


class PrioritizedReplay(Replay):
    def __init__(self,
                 learn_config,
                 env_config,
                 session_config):
        super().__init__(
            learn_config=learn_config,
            env_config=env_config,
            session_config=session_config
        )
        pass

    def default_config(self):
        conf = super().default_config()
        conf.update({
        })
        return conf

    def _insert(self, exp_dict):
        evicted = []
        return evicted

    def _sample(self, batch_size):
        return []

    def _evict(self, evict_size):
        return []

    def _start_sample_condition(self):
        pass

    def __len__(self):
        pass

