import random
from .base import Replay
from .aggregator import aggregate_torch


class UniformReplay(Replay):
    def __init__(self, *,
                 redis_client,
                 batch_size,
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
            **kwargs
        )
        # TODO: also drop Redis memory
        self._memory = []
        self._maxsize = memory_size
        self._sampling_start_size = sampling_start_size
        self._next_idx = 0

    def _insert(self, exp_dict):
        if self._next_idx >= len(self._memory):
            self._memory.append(exp_dict)
        else:
            self._memory[self._next_idx] = exp_dict
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _sample(self, batch_size, batch_i):
        indices = [random.randint(0, len(self._memory) - 1)
                   for _ in range(batch_size)]
        return [self._memory[i] for i in indices]

    def start_sample_condition(self):
        return len(self) > self._sampling_start_size

    def __len__(self):
        return len(self._memory)


class TorchUniformReplay(UniformReplay):
    def aggregate_batch(self, exp_list):
        return aggregate_torch(exp_list)
