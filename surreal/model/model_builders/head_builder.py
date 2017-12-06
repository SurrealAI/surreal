import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self,
                 input_spec,
                 output_spec=None):

        if type == 'discrete':
            raise NotImplementedError

        elif type == 'continuous':
            raise NotImplementedError

        elif type == 'gaussian':
            raise NotImplementedError

        elif type == 'scalar':
            raise NotImplementedError

        elif type == 'distributional':
            raise NotImplementedError

        else:
            raise ValueError('Unknown head type: {}'.format(type))

