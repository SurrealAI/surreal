import torch
import torch.nn as nn
from torch.autograd import Variable
import surreal.utils as U
import numpy as np

class ZFilter(U.Module):
    """
        Keeps historical average and std of inputs
        Whitens data and clamps to +/- 5 std
        Attributes:
            insize: state dimension
                required from input
            eps: tolerance value for computing Z-filter (whitening)
                default to 10
            use_cuda: whether GPU is used
                default to false
            running_sum: running sum of all previous states
                (Note, type is torch.cuda.FloatTensor or torch.FloatTensor)
            running_sumsq: sum of square of all previous states
                (Note, type is torch.cuda.FloatTensor or torch.FloatTensor)
            count: number of experiences accumulated
                (Note, type is torch.cuda.FloatTensor or torch.FloatTensor)
    """
    def __init__(self, in_size, eps=1e-2, use_cuda=False):
        """
        Constructor for ZFilter class
        Args:
            in_size: state dimension
            eps: tolerance value for computing Z-filter (whitening)
            use_cuda: whether GPU is used
        """
        super(ZFilter, self).__init__()
        self.eps = eps
        self.in_size = in_size

        # Keep some buffers for doing whitening. 
        self.register_buffer('running_sum', torch.zeros(in_size))
        self.register_buffer('running_sumsq', eps * torch.ones(in_size))
        self.register_buffer('count', torch.Tensor([eps]))

        # GPU option. Only used in learner. Pass in use_cuda flag from learner
        self.use_cuda = use_cuda
        if self.use_cuda:  
            self.running_sum   = self.running_sum.cuda()
            self.running_sumsq = self.running_sumsq.cuda()
            self.count         = self.count.cuda() 

    def z_update(self, x):
        """
            Count x into historical average, updates running sums and count
            Accepts batched input
            Args:
                x: input tensor to be kept in record. 
        """
        # only called in learner, so we can assume it has the correct type
        self.running_sum += torch.sum(x.data, dim=0)
        self.running_sumsq += torch.sum(x.data * x.data, dim=0)
        added_count = U.to_float_tensor(np.array([len(x)]))
        if self.use_cuda:
            added_count = added_count.cuda()
        self.count += added_count 

        print('mean: ', self.running_mean())
        print('std:', self.running_std())
        print('# exp accumulated:', self.count)
        print('---------------------')

    # define forward prop operations in terms of layers
    def forward(self, inputs):
        '''
            Whiten observation (inputs) to have zero-mean, unit variance.
            Also clamps output to be within 5 standard deviations
        '''
        # if True: return inputs
        running_mean = (self.running_sum / self.count)
        # running_std = ((self.running_sumsq / self.count) - running_mean.pow(2)).pow(0.5)
        # running_std += self.eps
        running_std = torch.clamp((self.running_sumsq / self.count \
                                  - running_mean.pow(2)).pow(0.5), min=self.eps)
        running_mean = Variable(running_mean)
        running_std = Variable(running_std)
        normed = torch.clamp((inputs - running_mean) / running_std, -5.0, 5.0)
        return normed

    def cuda(self):
        '''
            Converting all Tensor properties to be on GPU
        '''
        if self.use_cuda:  
            self.running_sum   = self.running_sum.cuda()
            self.running_sumsq = self.running_sumsq.cuda()
            self.count         = self.count.cuda() 
        else:
            print('.cuda() has no effect. Config set not to use GPU')

    def cpu(self):
        '''
            Converting all Tensor properties to be on CPU
        '''
        self.running_sum   = self.running_sum.cpu()
        self.running_sumsq = self.running_sumsq.cpu()
        self.count         = self.count.cpu() 

    def running_mean(self):
        '''
            returning the running obseravtion mean for Tensorplex logging
        '''
        running_mean = self.running_sum / self.count
        if self.use_cuda: 
            running_mean = running_mean.cpu()
        return running_mean.numpy()

    def running_std(self):
        '''
            returning the running standard deviation for Tensorplex Logging
        '''
        running_std = ((self.running_sumsq / self.count) 
                     - (self.running_sum / self.count).pow(2)).pow(0.5)
        if self.use_cuda:
            running_std = running_std.cpu() 
        return running_std.numpy()
        
    def running_square(self):
        '''
            returning the running square mean for Tensorplex Logging
        '''
        running_square = self.running_sumsq / self.count
        if self.use_cuda:
            running_square = running_square.cpu()
        return running_square.numpy()


'''
change tolerance to add small before start using
what works so far: fixing mean and variance
tests to run:
    1) blank test to get running std and mean √ 
    2) test adding small eps instead of clamping √
'''