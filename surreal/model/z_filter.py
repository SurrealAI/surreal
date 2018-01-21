import torch
import torch.nn as nn
from torch.autograd import Variable
import surreal.utils as U
import numpy as np

class ZFilter(U.Module):
    recurrent = False
    def __init__(self, in_size, eps=1e-2):
        """
        :param in_size: state dimension
        :param use_cuda: NOT IMPLEMENTED
        :param eps: tolerance value for computing Z-filter (whitening)
        """
        super(ZFilter, self).__init__()

        # Keep some buffers for doing whitening. 
        self.eps = eps
    
        self.register_buffer('running_sum', torch.zeros(in_size))
        self.register_buffer('running_sumsq', eps * torch.ones(in_size))
        self.register_buffer('count', torch.Tensor([eps]))

        self.in_size = in_size

        # Not Implemented
        self.use_cuda = False

        # mode
        self.train()

    # update internal state for whitening
    # Accepts batched input
    def z_update(self, x):
        self.running_sum += torch.sum(x.data, dim=0)
        self.running_sumsq += torch.sum(x.data * x.data, dim=0)
        self.count += U.to_float_tensor(np.array([len(x)]))

    # define forward prop operations in terms of layers
    def forward(self, inputs):
        running_mean = (self.running_sum / self.count)
        running_std = (torch.max((self.running_sumsq / self.count) - running_mean.pow(2), torch.Tensor([self.eps]))).pow(0.5)
        running_mean = Variable(running_mean)
        running_std = Variable(running_std)

        normed = torch.clamp((inputs - running_mean) / running_std, -5.0, 5.0)

        if self.training:
            self.z_update(inputs)

        return normed


    def running_mean(self):
        return (self.running_sum / self.count).numpy()

    def running_std(self):
        return (torch.max((self.running_sumsq / self.count) - (self.running_sum / self.count).pow(2), torch.Tensor([self.eps]))).pow(0.5).numpy()

    def running_square(self):
        return (self.running_sumsq / self.count).numpy()



