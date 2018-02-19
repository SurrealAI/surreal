import torch
import torch.nn as nn
from torch.autograd import Variable
import surreal.utils as U
import numpy as np

class ZFilter(U.Module):
    """
        Keeps historical average and std of inputs
        Whitens data and clamps to +/- 5 std
    """
    recurrent = False
    def __init__(self, in_size, eps=1e-2, use_cuda=False):
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

        # GPU option. Only used in learner. Needs to pass in a use_cuda flag from learner
        self.use_cuda = use_cuda
        if self.use_cuda:  
            self.running_sum   = self.running_sum.cuda()
            self.running_sumsq = self.running_sumsq.cuda()
            self.count         = self.count.cuda() 

    def z_update(self, x):
        """
            Count x into historical average
            Accepts batched input
        """
        self.running_sum += torch.sum(x.data, dim=0)
        self.running_sumsq += torch.sum(x.data * x.data, dim=0)
        added_count = U.to_float_tensor(np.array([len(x)]))
        if self.use_cuda:
            added_count = added_count.cuda()
        self.count += added_count 

    # define forward prop operations in terms of layers
    def forward(self, inputs):
        running_mean = (self.running_sum / self.count)
        running_std = (torch.clamp((self.running_sumsq / self.count) - running_mean.pow(2), min=self.eps)).pow(0.5)
        running_mean = Variable(running_mean)
        running_std = Variable(running_std)

        normed = torch.clamp((inputs - running_mean) / running_std, -5.0, 5.0)

        return normed


    def running_mean(self):
        if self.use_cuda: 
            return (self.running_sum / self.count).cpu().numpy() 
        else:
            return (self.running_sum / self.count).numpy()

    def running_std(self):
        if self.use_cuda:
            return (torch.clamp((self.running_sumsq / self.count) - running_mean.pow(2), min=self.eps)).pow(0.5).cpu().numpy() 
        else:
            return (torch.clamp((self.running_sumsq / self.count) - running_mean.pow(2), min=self.eps)).pow(0.5).numpy()

    def running_square(self):
        if self.use_cuda:
            return (self.running_sumsq / self.count).cpu().numpy()
        else: 
            return (self.running_sumsq / self.count).numpy()


