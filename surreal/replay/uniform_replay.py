import random
from .base import Replay
import surreal.utils as U


class UniformReplay(Replay):
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 index=0):
        """
        Args:
          memory_size: Max number of experience to store in the buffer.
            When the buffer overflows the old memories are dropped.
          sampling_start_size: min number of exp above which we will start sampling
        """
        super().__init__(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config,
            index=index
        )
        self._memory = []
        self.memory_size = self.learner_config.replay.memory_size
        self._next_idx = 0

    # def default_config(self):
    #     conf = super().default_config()
    #     conf.update({
    #         'memory_size': '_int_',
    #         'sampling_start_size': '_int_'
    #     })
    #     return conf

    def insert(self, exp_dict):
        if self._next_idx >= len(self._memory):
            self._memory.append(exp_dict)
        else:
            self._memory[self._next_idx] = exp_dict
        self._next_idx = (self._next_idx + 1) % self.memory_size

    def sample(self, batch_size):    
        indices = [random.randint(0, len(self._memory) - 1)
                   for _ in range(batch_size)]
        response = [self._memory[i] for i in indices]
        return response

    def evict(self):
        raise NotImplementedError  # TODO
        if evict_size > len(self._memory):
            evicted = self._memory
            self._memory = []
            self._next_idx = 0
            return evicted
        forward_space = len(self._memory) - self._next_idx
        if evict_size < forward_space:
            evicted = self._memory[self._next_idx:self._next_idx+evict_size]
            del self._memory[self._next_idx:self._next_idx+evict_size]
        else:
            evicted = self._memory[self._next_idx:]
            evict_from_left = evict_size - forward_space
            evicted += self._memory[:evict_from_left]
            del self._memory[self._next_idx:]
            del self._memory[:evict_from_left]
            self._next_idx -= evict_from_left
        assert len(evicted) == evict_size
        return evicted

    def start_sample_condition(self):
        return len(self) > self.learner_config.replay.sampling_start_size

    def __len__(self):
        return len(self._memory)
