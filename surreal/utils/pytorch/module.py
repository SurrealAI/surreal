"""
Hack torch.autograd.Variable:
from surreal.utils.pytorch import GpuVariable as Variable
"""
import os
import torch
import torch.nn as nn
from .gpu import get_scope_gpu
from .compute import torch_clip_norm
from surreal.utils.common import SaveInitArgs
from collections import OrderedDict
import numpy as np
import threading


def _net_or_parameters(net):
    if isinstance(net, torch.nn.Module):
        return net.parameters()
    else:
        return net


def set_requires_grad(net, requires_grad):
    """
    no gradients computed
    http://pytorch.org/docs/master/notes/autograd.html?highlight=volatile
    """
    for param in _net_or_parameters(net):
        param.requires_grad = requires_grad
    return net


def net_freeze(net):
    return set_requires_grad(net, requires_grad=False)


def net_unfreeze(net):
    return set_requires_grad(net, requires_grad=True)


def net_copy(net1, net2):
    """
    Assign net1's parameters to net2
    """
    net2.load_state_dict(net1.state_dict())


def net_clip_grad_value(net, clip_value):
    for param in _net_or_parameters(net):
        if param.grad is None:
            continue
        if clip_value < 0:
            raise ValueError('{} is not a valid gradient clip value.'.format(clip_value))
        param.grad.data.clamp_(-float(clip_value), float(clip_value))
    return net


def net_clip_grad_norm(net, clip, *, norm_type=2):
    """
    Unlike pytorch.nn.utils.net_clip_grad_norm,
    this function clips norm by every parameter
    original src:
    http://pytorch.org/docs/0.2.0/_modules/pytorch/nn/utils/clip_grad.html#net_clip_grad_norm
    """
    for param in _net_or_parameters(net):
        grad = param.grad
        if grad is None:
            continue
        torch_clip_norm(grad.data, clip, norm_type=norm_type, in_place=True)
    return net


def flatten_tensors(tensors):
    """
    Flatten tensors into a single contiguous 1D buffer
    https://github.com/pytorch/pytorch/blob/master/torch/_utils.py
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    numels = [tensor.numel() for tensor in tensors]
    size = sum(numels)
    offset = 0
    flat = tensors[0].new(size)
    for tensor, numel in zip(tensors, numels):
        flat.narrow(0, offset, numel).copy_(tensor, broadcast=False)
        offset += numel
    return flat


def unflatten_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors"""
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class Module(nn.Module, SaveInitArgs):
    """
    All models in Surreal should extend this module, not pytorch one
    """
    def __init__(self):
        super().__init__()
        self._gpu_ids = [-1]
        self._infinite_recursion_guard = False  # for nn.DataParallel calls

    def freeze(self):
        return net_freeze(self)

    def unfreeze(self):
        return net_unfreeze(self)

    def copy_from(self, other_net):
        net_copy(other_net, self)
        return self

    def copy_to(self, other_net):
        net_copy(self, other_net)
        return self

    def __call__(self, *args, **kwargs):
        """
        transfer to GPU before forward pass
        - if scope_gpu list has only one ID, send Module to that device
        - if scope_gpu list has more than one, automatically wrap with nn.DataParallel
        """
        if self._infinite_recursion_guard:
            return super().__call__(*args, **kwargs)
        scope_gpu_ids = get_scope_gpu()  # from torch_gpu_scope() context
        assert isinstance(scope_gpu_ids, list), 'internal error, must be list'
        self._gpu_ids = scope_gpu_ids
        if len(scope_gpu_ids) == 1:  # simply send to that device
            gpu_id = self._gpu_ids[0]
            if gpu_id >= 0:
                self.cuda(gpu_id)
            return super().__call__(*args, **kwargs)
        else:  # DataParallel
            parallel_wrapped = nn.DataParallel(self, device_ids=self._gpu_ids)
            parallel_wrapped.cuda()
            self._infinite_recursion_guard = True
            result = parallel_wrapped(*args, **kwargs)
            self._infinite_recursion_guard = False
            return result

    def clip_grad_value(self, clip):
        return net_clip_grad_value(self, clip)

    def clip_grad_norm(self, clip, norm_type=2):
        return net_clip_grad_norm(self, clip, norm_type=norm_type)

    def save(self, fname):
        save_dict = OrderedDict()
        # from meta class SaveInitArgs
        save_dict['init_args'] = self.init_args
        save_dict['torch'] = self.state_dict()
        torch.save(save_dict, fname)

    def load(self, fname):
        save_dict = torch.load(os.path.expanduser(fname))
        self.load_state_dict(save_dict['torch'])

    @classmethod
    def class_load(cls, fname):
        save_dict = torch.load(os.path.expanduser(fname))
        net = cls(**save_dict['init_args'])
        net.load_state_dict(save_dict['torch'])
        return net

    # Note: Currently unused and these code will not serialize everything needed to 
    # replicate module state. They will only serialize parameters.
    # 
    # def parameters_to_binary(self):
    #     params = [param.data for param in self.parameters()]
    #     flattened = flatten_tensors(params)
    #     return flattened.cpu().numpy().tostring()

    # def parameters_hash(self):
    #     return binary_hash(self.parameters_to_binary())

    # def parameters_from_binary(self, binary):
    #     """
    #     Assumes np.float32
    #     """
    #     buffer = np.fromstring(binary, dtype=np.float32)
    #     buffer = torch.from_numpy(buffer)
    #     params = [param.data for param in self.parameters()]
    #     new_params = unflatten_tensors(buffer, params)
    #     with self._forward_lock:
    #         for p, n in zip(params, new_params):
    #             p.copy_(n)

    def clone(self, no_grad=True):
        """
        The target Q network should not do any backprop
        """
        qcopy = type(self)(**self.init_args).copy_from(self)
        if no_grad:
            qcopy.freeze()
        return qcopy

    @property
    def is_cuda(self):
        return self._gpu_ids[0] >= 0
