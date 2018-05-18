import torch
import torch.nn as nn
import surreal.utils as U
import numpy as np
import torchx.nn as nnx

class ZFilter(nnx.Module):
    """
        Keeps historical average and std of inputs
        Whitens data and clamps to +/- 5 std
        Attributes:
            in_size: state dimension
                required from input
            eps: tolerance value for computing Z-filter (whitening)
                default to 10
            running_sum: running sum of all previous states
                (Note, type is torch.cuda.FloatTensor or torch.FloatTensor)
            running_sumsq: sum of square of all previous states
                (Note, type is torch.cuda.FloatTensor or torch.FloatTensor)
            count: number of experiences accumulated
                (Note, type is torch.cuda.FloatTensor or torch.FloatTensor)
    """
    def __init__(self, obs_spec, eps=1e-5):
        """
        Constructor for ZFilter class
        Args:
            obs_spec: nested dictionary of observation space spec. see doc
            eps: tolerance value for computing Z-filter (whitening)
        """
        super(ZFilter, self).__init__()
        self.eps = eps
        self.obs_spec = obs_spec

        in_size = 0
        for key in self.obs_spec['low_dim'].keys():
            in_size += self.obs_spec['low_dim'][key][0]
        self.in_size = in_size

        # Keep some buffers for doing whitening. 
        self.register_buffer('running_sum', torch.zeros(in_size))
        self.register_buffer('running_sumsq', eps * torch.ones(in_size))
        self.register_buffer('count', torch.tensor([eps], dtype=torch.float32))

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
        self.running_sum += torch.sum(x, dim=0)
        self.running_sumsq += torch.sum(x * x, dim=0)
        self.count += float(len(x))

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
        normed = torch.clamp((inputs - running_mean) / running_std, -5.0, 5.0)
        normed = normed.view(input_shape)
        return normed

    def running_mean(self):
        '''
            returning the running obseravtion mean for Tensorplex logging
            Returns:
                numpy array of current running observation mean
        '''
        running_mean = self.running_sum / self.count
        return running_mean.cpu().numpy()

    def running_std(self):
        '''
            returning the running standard deviation for Tensorplex Logging
            Returns:
                numpy array of running standard deviation
        '''
        running_std = ((self.running_sumsq / self.count) 
                     - (self.running_sum / self.count).pow(2)).pow(0.5)
        return running_std.cpu().numpy()
        
    def running_square(self):
        '''
            returning the running square mean for Tensorplex Logging
            Returns:
                running square mean
        '''
        running_square = self.running_sumsq / self.count
        return running_square.cpu().numpy()
