import time
import numpy as np
import random
from .base import Replay
from .aggregator import aggregate_torch


class DummyReplay(Replay):
    def __init__(self, *args, sampling_start_size, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampling_start_size = sampling_start_size
        self._memory = {}

    def _insert(self, exp_dict):
        print('INSERT', exp_dict['exp_pointer'])
        time.sleep(0.2)
        self._memory[len(self._memory)] = exp_dict

    def _sample(self, batch_size, batch_i):
        samps = []
        print('SAMPLE START total memory', len(self._memory), 'batch_i', batch_i)
        assert self.start_sample_condition()
        indices = [random.randint(0, len(self._memory) - 1)
                   for _ in range(batch_size)]
        memkeys = list(self._memory.keys())
        for i in indices:
            time.sleep(.3)
            samps.append(self._memory[memkeys[i]])
        print('SAMPLE DONE')
        return samps

    def _evict(self, evict_size):
        evict_keys = list(self._memory.keys())[:evict_size]
        print('EVICT START')
        time.sleep(1)
        exps = []
        for k in evict_keys:
            exps.append(self._memory.pop(k))
        print('EVICT DONE:', [exp['exp_pointer'] for exp in exps])
        return exps

    def start_sample_condition(self):
        return len(self._memory) > self._sampling_start_size

    def aggregate_batch(self, exp_list):
        return aggregate_torch(exp_list)
        # aggreg = {}
        # for key in exp_list[0]:
        #     aggreg[key] = []
        # for exp in exp_list:
        #     for key in exp:
        #         aggreg[key].append(exp[key])
        # return aggreg
