import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class LinearWithMinLR(_LRScheduler):
    def __init__(self, optimizer, 
                       num_steps, 
                       init_lr,
                       update_freq=1, 
                       min_lr=1e-5, 
                       last_epoch=-1):

        self.num_steps = num_steps
        self.update_freq = update_freq
        self.min_lr = min_lr
        total_updates = int(num_steps / update_freq)
        self.anneal_quantity = (init_lr - min_lr) / total_updates
        super(LinearWithMinLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.update_freq == 0:
            return [max(self.min_lr, base_lr - self.anneal_quantity * self.last_epoch) 
                                        for i, base_lr in enumerate(self.base_lrs)]
        else:
            return self.base_lrs