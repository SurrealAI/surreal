import random
from .base import Replay
from .aggregator import torch_aggregate


class FIFOReplay(Replay):
    def __init__(self, *,
                 redis_client,
                 batch_size,
                 obs_spec,
                 action_spec,
                 memory_size,
                 sampling_start_size,
                 **kwargs):
        """
        Args:
          memory_size: Max number of experience to store in the buffer.
            When the buffer overflows the old memories are dropped.
          sampling_start_size: min number of exp above which we will start sampling
        """
        super().__init__(
            redis_client=redis_client,
            batch_size=batch_size,
            obs_spec=obs_spec,
            action_spec=action_spec,
            **kwargs
        )
        self._maxsize = memory_size
        self._sampling_start_size = sampling_start_size
        self._next_idx = 0

    def _insert(self, exp_dict):
        raise NotImplementedError

    def _sample(self, batch_size, batch_i):
        raise NotImplementedError

    def _evict(self, *args, **kwargs):
        return []

    def start_sample_condition(self):
        raise NotImplementedError

