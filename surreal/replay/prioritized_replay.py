# Adapted from https://github.com/openai/baselines

import numpy as np
import random

from segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedReplay(Replay):
    
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config):
        """
        Create Prioritized Replay buffer.
        :param size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        :param alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config
        )

        self._alpha = self.replay_config.alpha
        assert self._alpha > 0

        self._memory = []
        self.memory_size = self.replay_config.memory_size
        it_capacity = 1
        while it_capacity < self.memory_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def default_config(self):
        conf = super().default_config()
        conf.update({
            'memory_size': '_int_',
            'sampling_start_size': '_int_',
            'alpha': '_float_',
        })
        return conf

    def insert(self, exp_dict):
        """
        Adds experience to the replay buffer as usual, but also
        intiialize the priority of the new experience.
        """
        with self.insert_time.time():
            if self._next_idx >= len(self._memory):
                self._memory.append(exp_dict)
            else:
                self._memory[self._next_idx] = exp_dict
            self._next_idx = (self._next_idx + 1) % self.memory_size

            idx = self._next_idx
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha

    def sample(self, batch_size, beta=0):
        """
        WARNING: This function does not make deep copies of the tuple experiences.
                 This means that if any objects in the experiences are modified,
                 the contents of the replay buffer memory will also be modified,
                 so be careful!!!
        Sample a batch of experiences, along with their importance weights, and the 
        indices of the sampled experiences in the buffer.
        :param batch_size: int
            How many transitions to sample.
        :param beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        :return experience_batch: List
            List of tuples, length batch_size, corresponding to the experiences sampled.
        :return weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        :return indices: np.array
            Array of shape (batch_size,) and dtype np.int32
            indices in buffer of sampled experiences
        """
        with self.sample_time.time():
            assert beta >= 0

            # sample the experiences proportional to their priorities
            indices = self._sample_proportional(batch_size)
            response = [self._storage[idx] for idx in indices]

            # compute importance weights for the experiences to correct for distribution shift
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in indices:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)

        # return response, weights, indices
        return response

    def _sample_proportional(self, batch_size):
        """
        This is a helper function to sample expriences with probabilities 
        proportional to their priorities.
        Returns a list of indices.
        """
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, indices, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index indices[i] in buffer
        to priorities[i].
        :param indices: [int]
            List of indices of sampled transitions
        :param priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled indices denoted by
            variable `indices`.
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def evict(self):
        raise NotImplementedError # TODO
        # if evict_size > len(self._memory):
        #     evicted = self._memory
        #     self._memory = []
        #     self._next_idx = 0
        #     return evicted
        # forward_space = len(self._memory) - self._next_idx
        # if evict_size < forward_space:
        #     evicted = self._memory[self._next_idx:self._next_idx+evict_size]
        #     del self._memory[self._next_idx:self._next_idx+evict_size]
        # else:
        #     evicted = self._memory[self._next_idx:]
        #     evict_from_left = evict_size - forward_space
        #     evicted += self._memory[:evict_from_left]
        #     del self._memory[self._next_idx:]
        #     del self._memory[:evict_from_left]
        #     self._next_idx -= evict_from_left
        # assert len(evicted) == evict_size
        # return evicted

    def start_sample_condition(self):
        return len(self) > self.replay_config.sampling_start_size

    def __len__(self):
        return len(self._memory)
