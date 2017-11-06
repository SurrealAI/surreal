import random


class UniformReplay():
    def __init__(self,
                 memory_size,
                 **kwargs):
        """
        Args:
          memory_size: Max number of experience to store in the buffer.
            When the buffer overflows the old memories are dropped.
          sampling_start_size: min number of exp above which we will start sampling
        """
        self._memory = []
        self._maxsize = memory_size
        self._sampling_start_size = 0
        self._next_idx = 0

    def _insert(self, exp_dict):
        evicted = []
        if self._next_idx >= len(self._memory):
            self._memory.append(exp_dict)
        else:
            evicted.append(self._memory[self._next_idx])
            self._memory[self._next_idx] = exp_dict
        self._next_idx = (self._next_idx + 1) % self._maxsize
        return evicted

    def _sample(self, batch_size, batch_i):
        indices = [random.randint(0, len(self._memory) - 1)
                   for _ in range(batch_size)]
        return [self._memory[i] for i in indices]

    def _evict(self, evict_size):
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
        return len(self) > self._sampling_start_size

    def __len__(self):
        return len(self._memory)


replay=UniformReplay(10)
for i in range(16):
    print(replay._insert({i}))
print(replay._memory)
replay._evict(2)
print(replay._memory)
for i in range(100,107):
    replay._insert({i})
print(replay._memory)
