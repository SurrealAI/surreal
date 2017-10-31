import random
import torch
import numpy as np
from easydict import EasyDict
import surreal.utils as U
from surreal.utils.torch_util import GpuVariable as Variable
from surreal.comm import RedisClient
from .base import Replay


class UniformReplay(Replay):
    def __init__(
        self,
        redis_client,
        memory_size,
        sampling_start_size,
        batch_size,
        download_queue_size,
        name='replay'
    ):
        """
        Args:
          memory_size: Max number of experience to store in the buffer.
            When the buffer overflows the old memories are dropped.
          sampling_start_size: min number of exp above which we will start sampling
        """
        super().__init__(
            redis_client=redis_client,
            batch_size=batch_size,
            download_queue_size=download_queue_size,
            name=name
        )
        # TODO: also drop Redis memory
        self._memory = []
        self._maxsize = memory_size
        self._sampling_start_size = sampling_start_size
        self._next_idx = 0

    def insert(self, exp_dict):
        if self._next_idx >= len(self._memory):
            self._memory.append(exp_dict)
        else:
            self._memory[self._next_idx] = exp_dict
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size, batch_i):
        indices = [random.randint(0, len(self._memory) - 1)
                   for _ in range(batch_size)]
        return [self._memory[i] for i in indices]

    def start_sample_condition(self):
        return len(self) > self._sampling_start_size

    def __len__(self):
        return len(self._memory)


class TorchUniformReplay(UniformReplay):
    def _obs_concat(self, obs_list):
        # convert uint8 to float32, if any
        return Variable(U.to_float_tensor(np.stack(obs_list)))

    def aggregate_batch(self, exp_list):
        obses0, actions, rewards, obses1, dones = [], [], [], [], []
        for exp in exp_list:
            obses0.append(np.array(exp['obses'][0], copy=False))
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            obses1.append(np.array(exp['obses'][1], copy=False))
            dones.append(float(exp['done']))
        return EasyDict(
            obses=[self._obs_concat(obses0), self._obs_concat(obses1)],
            actions=Variable(torch.LongTensor(actions).unsqueeze(1)),
            rewards=Variable(torch.FloatTensor(rewards).unsqueeze(1)),
            dones=Variable(torch.FloatTensor(dones).unsqueeze(1)),
        )
