import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .data import shape


def torch_median(t):
    """
    Find median of entire tensor or Variable
    """
    return TC.to_tensor(t).view(-1).median(dim=0)[0][0]


def torch_median_abs(t):
    return torch_median(t.abs())


def torch_ones_like(tensor):
    s = shape(tensor)
    assert s is not None
    return torch.ones(s)


def torch_zeros_like(tensor):
    s = shape(tensor)
    assert s is not None
    return torch.zeros(s)


def normalize_feature(feats):
    """
    Normalize the whole dataset as one giant feature matrix
    """
    mean = feats.mean(0).expand_as(feats)
    std = feats.std(0).expand_as(feats)
    return (feats - mean) / std


def torch_where(cond, x1, x2):
    """
    Similar to np.where and tf.where
    """
    cond = cond.type_as(x1)
    return cond * x1 + (1 - cond) * x2


def huber_loss_per_element(x, y=None, delta=1.0):
    """
    Args:
        if y is not None, compute huber_loss(x - y)
    """
    if y is not None:
        x = x - y
    x_abs = x.abs()
    return torch_where(x_abs < delta,
                       0.5 * x * x,
                       delta * (x_abs - 0.5 * delta))


def torch_norm(tensor, norm_type=2):
    """
    Supports infinity norm
    """
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        return tensor.abs().max()
    else:
        return tensor.norm(norm_type)


def torch_clip_norm(tensor, clip, norm_type=2, in_place=False):
    """
    original src:
    http://pytorch.org/docs/0.2.0/_modules/pytorch/nn/utils/clip_grad.html#net_clip_grad_norm
    """
    norm = torch_norm(tensor, norm_type)
    clip_coef = clip / (norm + 1e-6)
    if clip_coef < 1:
        if in_place:
            tensor.mul_(clip_coef)
        else:
            tensor = tensor * clip_coef
    return tensor


