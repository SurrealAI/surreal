import os
import torch
import torch.nn as nn
import math
from surreal.utils.checkpoint import *
import surreal.utils as U


class MyNet(nn.Module):
    def __init__(self, width1, width2):
        super().__init__()
        self.fc1 = nn.Linear(width1, width2)
        self.fc2 = nn.Linear(width2, 3)

    def set_weight(self, x):
        self.fc1.weight.data.fill_(x)
        self.fc1.bias.data.fill_(-x)
        self.fc2.weight.data.fill_(x * 10)
        self.fc2.bias.data.fill_(-x * 10)

    def show_weight(self):
        return list(map(float, map(torch.mean,
                        [self.fc1.weight.data,
                         self.fc1.bias.data,
                         self.fc2.weight.data,
                         self.fc2.bias.data])))

class Learner():
    def __init__(self, lr, eps, width1, width2):
        self.lr = lr
        self.eps = eps
        self.mylist = [1, 2, 3]
        self.net1 = MyNet(width1, width2)
        self.net2 = MyNet(width2, width1)
        self._count = 0
        self.checkpoint = PeriodicCheckpoint(
            folder='~/Temp/' + FOLDER,
            name='learner',
            period=3,
            tracked_obj=self,
            tracked_attrs=['lr', 'net1', 'eps', 'mylist', 'net2'],
            keep_history=5,
            keep_best=3,
        )

    def get_score(self):
        return math.sin(self._count * math.pi/6) * math.exp(self._count/5)

    def show_weight(self):
        print('net1', self.net1.show_weight())
        print('net2', self.net2.show_weight())

    def train(self, callback):
        while True:
            self._count += 1
            c = self._count
            steps = c * 1000
            self.eps -= 0.01
            self.lr *= 0.9
            for i in range(3):
                self.mylist[i] += 0.01
            self.net1.set_weight(c)
            self.net2.set_weight(c * 2)
            self.checkpoint.save(self.get_score(),
                                 global_steps=steps,
                                 reload_metadata=True)
            callback()

LR = 0.1
EPS = 0.99
WIDTH1 = 13
WIDTH2 = 17
FOLDER = 'otherckpt2'

learner_save = Learner(LR, EPS, WIDTH1, WIDTH2)
learner_load = Learner(LR, EPS, WIDTH1, WIDTH2)

checkpoint = Checkpoint(
    folder='~/Temp/' + FOLDER,
    name='learner',
    tracked_obj=learner_load,
)

def callback():
    print('='*70)
    print('learner_save')
    learner_save.show_weight()
    print('BEFORE load')
    learner_load.show_weight()
    # checkpoint.restore(0, mode='history', reload_metadata=True, check_ckpt_exists=True)
    print('AFTER load')
    learner_load.show_weight()
    input('continue ...')


if 0:
    learner_save.train(callback)
else:  # demo restore
    for i in range(11):
        ret = checkpoint.restore(i, mode='history', reload_metadata=True, check_ckpt_exists=1,
                                 restore_folder='~/Temp/otherckpt')
        print(ret)
        if ret:
            learner_load.show_weight()
    # checkpoint.restore_full_name('learner.best-16000.ckpt')
