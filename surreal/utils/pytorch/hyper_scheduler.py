# Custom learning rate schedulers. details see package torch.optim.lr_scheduler 
# or docs http://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class LinearWithMinLR(_LRScheduler):
    '''
        Subclass of _LRScheduler that linearly anneals learning rate every
            update_freq steps until a specified minimum learning rate is reached
        Attributes:
            num_steps: number of total steps to anneal
            init_lr: initial learning rate
            update_freq: anneals learning rate every X steps
                Note: if this value is below 1, then no annealing is used
            min_lr: minimum learning rate
            last_epoch: tracks current step
        Member functions:
            get_lr: return list of scheduled learning rate
            step: (see parent class) function that is called to apply learning
                rate change and increases last_epoch
    '''
    def __init__(self, optimizer, 
                       num_steps, 
                       update_freq=1, 
                       min_lr=1e-5, 
                       last_epoch=-1):

        self.num_steps = num_steps
        self.update_freq = update_freq
        self.min_lr = min_lr
        super(LinearWithMinLR, self).__init__(optimizer, last_epoch)

        if self.update_freq < 1:
            self.anneal_quantity = None
        else:
            total_updates = int(num_steps / update_freq)
            self.anneal_quantity = [(lr - min_lr)/total_updates for lr in self.base_lrs]

    def get_lr(self):
        '''
            Anneals current learning rate unless it is specified minimum learning rate
            Returns:
                a list of updated learning rate. one for each parameter group
        '''
        if self.update_freq < 1 or self.last_epoch < 1: return self.base_lrs
        num_updated = int(self.last_epoch / self.update_freq)
        return [max(self.min_lr, base_lr - self.anneal_quantity[i] * num_updated)
                        for i, base_lr in enumerate(self.base_lrs)]




