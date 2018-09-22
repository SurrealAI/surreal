import random
from collections import deque
from .base import Replay


class FIFOReplay(Replay):
    """
    WARNING: you must set session_config.replay:
    - max_puller_queue: to a very small number, like 1
    - max_prefetch_queue: to 1
    session_config.sender:
    - flush_iteration: to a small number
    """
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 index=0):
        super().__init__(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config,
            index=index,
        )
        self.batch_size = self.learner_config.replay.batch_size
        self.memory_size = self.learner_config.replay.memory_size
        self._memory = deque(maxlen=self.memory_size+3)  # + 3 for a gentle buffering
        assert self.session_config.replay.max_puller_queue <= 10
        assert self.session_config.replay.max_prefetch_queue == 1 
        assert not self.session_config.sender.flush_time
        assert self.session_config.sender.flush_iteration <= 10


    def insert(self, exp_tuple):
        self._memory.append(exp_tuple)

    def sample(self, batch_size):
        assert batch_size <= self.memory_size
        return [self._memory.popleft() for _ in range(batch_size)]

    def evict(self):
        raise NotImplementedError('no support for eviction in FIFO mode')

    def start_sample_condition(self):
        return len(self._memory) >= self.batch_size

    def __len__(self):
        return len(self._memory)

