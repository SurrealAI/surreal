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
    def __init__(self, obs_spec, input_config, pixel_input = False, eps=1e-2, use_cuda=False):
        """
        Constructor for ZFilter class
        Args:
            in_size: state dimension
            eps: tolerance value for computing Z-filter (whitening)
            use_cuda: whether GPU is used
        """
        super(ZFilter, self).__init__()
        self.eps = eps
        self.obs_spec = obs_spec
        self.input_config = input_config

        in_size = 0
        for key in self.obs_spec['low_dim'].keys():
            in_size += self.obs_spec['low_dim'][key][0]

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
        if x is None: return

        # only called in learner, so we can assume it has the correct type
        if len(x.size()) == 3: x = x.view(-1, self.in_size)
        self.running_sum += torch.sum(x.data, dim=0)
        self.running_sumsq += torch.sum(x.data * x.data, dim=0)
        added_count = U.to_float_tensor(np.array([len(x)]))
        if self.use_cuda:
            added_count = added_count.cuda()
        self.count += added_count 

    def forward(self, inputs):
        '''
            Whiten observation (inputs) to have zero-mean, unit variance.
            Also clamps output to be within 5 standard deviations
            Args:
                inputs -- batched observation input. batch size at dim 0
            Returns:
                0 mean std 1 weightened batch of observation
        '''
        if inputs is None: return None

        input_shape = inputs.size()
        assert len(input_shape) >= 2
        inputs = inputs.view(-1, input_shape[-1])

        running_mean = (self.running_sum / self.count)
        running_std = torch.clamp((self.running_sumsq / self.count \
                                  - running_mean.pow(2)).pow(0.5), min=self.eps)
        running_mean = Variable(running_mean)
        running_std = Variable(running_std)
        normed = torch.clamp((inputs - running_mean) / running_std, -5.0, 5.0)
        normed = normed.view(input_shape)
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
            Returns:
                numpy array of current running observation mean
        '''
        running_mean = self.running_sum / self.count
        if self.use_cuda: 
            running_mean = running_mean.cpu()
        return running_mean.numpy()

    def running_std(self):
        '''
            returning the running standard deviation for Tensorplex Logging
            Returns:
                numpy array of running standard deviation
        '''
        running_std = ((self.running_sumsq / self.count) 
                     - (self.running_sum / self.count).pow(2)).pow(0.5)
        if self.use_cuda:
            running_std = running_std.cpu() 
        return running_std.numpy()
        
    def running_square(self):
        '''
            returning the running square mean for Tensorplex Logging
            Returns:
                running square mean
        '''
        running_square = self.running_sumsq / self.count
        if self.use_cuda:
            running_square = running_square.cpu()
        return running_square.numpy()
