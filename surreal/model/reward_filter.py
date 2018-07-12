import torch
import numpy as np
import torchx.nn as nnx

class RewardFilter(nnx.Module):
    """
        Keeps historical average of rewards
        Attributes:
            eps: tolerance value for computing reward filter (whitening)
                default to 10
            running_sum: running sum of all previous rewards
                (Note, type is torch.cuda.FloatTensor or torch.FloatTensor)
            running_sumsq: sum of square of all previous states
                (Note, type is torch.cuda.FloatTensor or torch.FloatTensor)
            count: number of experiences accumulated
                (Note, type is torch.cuda.FloatTensor or torch.FloatTensor)
    """
    def __init__(self, eps=1e-5):
        """
            Constructor for RewardFilter class
            Args:
                eps: tolerance value for computing reward filter (whitening)
        """
        super(RewardFilter, self).__init__()

        self.eps = eps

        # Keep some buffers for doing whitening. 
        self.register_buffer('count', torch.tensor(eps, dtype=torch.float32))
        self.register_buffer('running_sum', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('running_sumsq', torch.tensor(0.0, dtype=torch.float32))

    def update(self, x):
        """
            Count x into historical average, updates running sums and count
            Accepts batched input
            Args:
                x: input tensor to be kept in record. 
        """
        self.count += float(np.prod(x.size()))
        self.running_sum += x.sum()
        self.running_sumsq = (x * x).sum()

    def forward(self, inputs):
        '''
            Whiten reward to have zero-mean, unit variance.
            Also clamps output to be within 5 standard deviations
            Args:
                inputs -- batched reward input. batch size at dim 0
            Returns:
                0 mean std 1 weightened batch of observation
        '''
        mean = (self.running_sum / self.count)
        std  = torch.clamp((self.running_sumsq / self.count \
                                  - mean.pow(2)).pow(0.5), min=self.eps)
        normed = torch.clamp((inputs - mean) / std, -5.0, 5.0)
        return normed

    def reward_mean(self):
        '''
            Outputs the current reward mean
        '''
        return (self.running_sum / self.count).item()
