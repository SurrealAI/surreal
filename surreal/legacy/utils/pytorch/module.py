import os
import torch
from .gpu import get_scope_gpu
from .compute import torch_clip_norm
from surreal.utils.common import SaveInitArgs
from surreal.utils.serializer import binary_hash
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


def net_clip_grad_value(net, clip):
    for param in _net_or_parameters(net):
        if param.grad is None:
            continue
        param.grad.data.clamp_(clip)
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


class Module(torch.nn.Module, SaveInitArgs):
    """
    All models in Surreal should extend this module, not pytorch one
    """
    def __init__(self):
        super().__init__()
        # Locks forward prop to avoid race condition with parameter server recv
        self._forward_lock = threading.Lock()
        self.gpu_index = -1

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

    def scoped_cuda(self):
        """
        Unlike the builtin .cuda(), this call sends to the device specified
        in the `with torch_gpu_scope` contextk
        """
        scope_gpu_index = get_scope_gpu()  # from torch_gpu_scope() context
        if scope_gpu_index >= 0 and self.gpu_index < 0:
            self.gpu_index = scope_gpu_index
            self.cuda(self.gpu_index)
        return self

    def __call__(self, *args, **kwargs):
        """
        transfer to GPU before forward pass
        """
        self.scoped_cuda()
        with self._forward_lock:
            return super().__call__(*args, **kwargs)

    def clip_grad_value(self, clip):
        return net_clip_grad_value(self, clip)

    def clip_grad_norm(self, clip, norm_type=2):
        return net_clip_grad_norm(self, clip, norm_type=norm_type)

    def save(self, fname):
        save_dict = OrderedDict()
        # from meta class SaveInitArgs
        with self._forward_lock:
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

    def parameters_to_binary(self):
        params = [param.data for param in self.parameters()]
        flattened = flatten_tensors(params)
        return flattened.cpu().numpy().tostring()

    def parameters_hash(self):
        return binary_hash(self.parameters_to_binary())

    def parameters_from_binary(self, binary):
        """
        Assumes np.float32
        """
        buffer = np.fromstring(binary, dtype=np.float32)
        buffer = torch.from_numpy(buffer)
        params = [param.data for param in self.parameters()]
        new_params = unflatten_tensors(buffer, params)
        with self._forward_lock:
            for p, n in zip(params, new_params):
                p.copy_(n)

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
        return self.gpu_index >= 0
