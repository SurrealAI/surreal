import torch
import torch.nn as nn
from torch.autograd import Variable
import surreal.utils as U
import numpy as np

def to_tensor(np_array, cuda=True):
    if cuda:
        return torch.from_numpy(np_array).cuda()
    else:
        return torch.from_numpy(np_array)


class ZFilter(nn.Module):
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
    def z_update(self, x):
        self.running_sum += torch.mean(x.data, dim=0)
        self.running_sumsq += torch.mean(x.data * x.data, dim=0)
        self.count += to_tensor(np.array([len(x)]), cuda=self.use_cuda).float()

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