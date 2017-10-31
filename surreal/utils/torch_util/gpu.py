import contextlib
import torch
from torch.autograd import Variable
from surreal.utils.common import noop_context
import sys


_PYTORCH_GPU_ = []


gpu_count = torch.cuda.device_count


def get_scope_gpu():
    return _PYTORCH_GPU_[-1] if _PYTORCH_GPU_ else -1


def _data_to_cuda(data):
    if data is None:
        return None
    gpu = get_scope_gpu()
    assert isinstance(gpu, int)
    if gpu >= 0:
        return data.cuda(gpu)
    else:
        return data


class GpuVariable(Variable):
    """
    Hack torch_util variable to auto transfer to GPU
    Have the following line at the top:

    from ml_utils import GpuVariable as Variable
    """
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        # WARNING: don't do this before super(). Due to cwrap issues,
        # super() will not capture the updated `data` local variable
        self.data = _data_to_cuda(self.data)


class GpuParameter(torch.nn.Parameter):
    """
    Hack torch_util variable to auto transfer to GPU
    Have the following line at the top:

    from ml_utils import GpuVariable as Variable
    """
    def __new__(cls, data=None, *args, **kwargs):
        data = _data_to_cuda(data)
        return torch.nn.Parameter.__new__(cls, data, *args, **kwargs)


@contextlib.contextmanager
def torch_gpu_scope(gpu=0, override_parent=True):
    """
    Magically force all GpuVariables to transfer to CUDA
    Args:
        gpu: -1 for CPU, otherwise the 0-based index of the GPU device
        override_parent: True to override the device in parent scope.
    """
    global _PYTORCH_GPU_
    count = torch.cuda.device_count()
    if count == 0 and gpu >= 0:
        print('WARNING: no GPU found, fall back to CPU.', file=sys.stderr)
        device_ctx = noop_context
        gpu = -1
    elif gpu >= count:
        raise RuntimeError('Not enough GPUs: only {} available'.format(count))
    else:
        device_ctx = torch.cuda.device
        if not override_parent and _PYTORCH_GPU_:
            # inherit from parent scope
            gpu = _PYTORCH_GPU_[-1]

    with device_ctx(gpu):
        _PYTORCH_GPU_.append(gpu)
        yield
        # restore
        _PYTORCH_GPU_.pop()

