import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
import torch.nn.init as init
from .gpu import GpuVariable as Variable


def zero_init(tensor):
    """
    Zero initialization method
    """
    if isinstance(tensor, torch.autograd.Variable):
        zero_init(tensor.data)
        return tensor
    return tensor.zero_()


def is_conv_layer(layer, dim=None):
    if dim is None:
        cls = _ConvNd
    elif dim == 1:
        cls = nn.Conv1d
    elif dim == 2:
        cls = nn.Conv2d
    elif dim == 3:
        cls = nn.Conv3d
    return isinstance(layer, cls)


def conv_fc_init(layer,
                 weight_init=init.xavier_uniform,
                 bias_init=zero_init):
    """
    Initialize a layer's filter weights by xavier and bias weights to zero
    The layer can be either nn.ConvNd or nn.Linear
    """
    if isinstance(layer, (list, nn.ModuleList)):
        return type(layer)([conv_fc_init(l,
                                         weight_init=weight_init,
                                         bias_init=bias_init)
                            for l in layer])
    assert is_conv_layer(layer) or isinstance(layer, nn.Linear)
    weight_init(layer.weight)
    bias_init(layer.bias)
    return layer


def flatten_conv(x):
    """
    https://discuss.pytorch.org/t/runtimeerror-input-is-not-contiguous/930/4
    `.contiguous()` copies the tensor if the data isn't contiguous
    """
    return x.contiguous().view(x.size(0), -1)


def global_avg_pool(x):
    """
    https://arxiv.org/pdf/1312.4400.pdf
    Average each feature map HxW to one number.
    """
    N, C, H, W = x.size()
    return x.view(N, C, H * W).mean(dim=2).squeeze(dim=2)


def global_max_pool(x):
    N, C, H, W = x.size()
    # pytorch.max returns a tuple of (max, indices)
    return x.view(N, C, H * W).max(dim=2)[0].squeeze(dim=2)


class FlattenConv(nn.Module):
    def forward(self, x):
        return flatten_conv(x)


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return global_avg_pool(x)


class GlobalMaxPool(nn.Module):
    def forward(self, x):
        return global_max_pool(x)

