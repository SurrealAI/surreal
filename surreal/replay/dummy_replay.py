import time
from .base import Replay


class DummyReplay(Replay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = {}

    def insert(self, exp_dict):
        print('INSERT', exp_dict)
        time.sleep(0.5)
        self.memory[len(self.memory)] = exp_dict

    def sample(self, batch_size, batch_i):
        samps = []
        print('SAMPLE total size', len(self.memory), 'batch_i', batch_i)
        for i in self.memory:
            time.sleep(.3)
            if i % 3 == 0:
                samps.append(self.memory[i])
        return samps

    def start_sample_condition(self):
        return len(self.memory) > 10

    def aggregate_batch(self, exp_list):
        aggreg = {}
        for key in exp_list[0]:
            aggreg[key] = []
        for exp in exp_list:
            for key in exp:
                aggreg[key].append(exp[key])
        return aggreg
