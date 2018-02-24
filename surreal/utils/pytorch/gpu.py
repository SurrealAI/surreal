import contextlib
import torch
from torch.autograd import Variable
from surreal.utils.common import noop_context
import sys


# list of device id list
# if the list has more than 1 GPU, it's meant for torch.nn.DataParallel
_PYTORCH_GPU_ = []


gpu_count = torch.cuda.device_count


def get_scope_gpu():
    return _PYTORCH_GPU_[-1] if _PYTORCH_GPU_ else [-1]


def _data_to_cuda(data):
    if data is None:
        return None
    gpu = get_scope_gpu()
    assert isinstance(gpu, list), 'internal error, get_scope_gpu() must be list'
    gpu = gpu[0]
    assert isinstance(gpu, int), 'internal error'
    if gpu >= 0:
        return data.cuda(gpu)
    else:
        return data


class GpuVariable(Variable):
    """
    Hack pytorch variable to auto transfer to GPU
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
    Hack torch variable to auto transfer to GPU
    Have the following line at the top:

    from ml_utils import GpuVariable as Variable
    """
    def __new__(cls, data=None, *args, **kwargs):
        data = _data_to_cuda(data)
        return torch.nn.Parameter.__new__(cls, data, *args, **kwargs)


@contextlib.contextmanager
def torch_gpu_scope(gpu, override_parent=True):
    """
    Magically force all GpuVariables to transfer to CUDA
    Args:
        gpu: -1 for CPU, otherwise the 0-based index of the GPU device
            if gpu is a list, use nn.DataParallel wrapper for utils.pytorch.Module
            Other variables will be sent to the first device in the list.
        override_parent: True to override the device in parent scope.
    """
    global _PYTORCH_GPU_
    count = torch.cuda.device_count()
    if isinstance(gpu, int):
        gpu = [gpu]
    else:
        assert isinstance(gpu, list)
        # https://github.com/pytorch/pytorch/issues/1280
        assert gpu[0] == 0, \
            'for multiGPU training, the first GPU ID must be 0. ' \
            'You can set CUDA_VISIBLE_DEVICES env variable to work around.'
    if count == 0 and gpu[0] >= 0:
        print('WARNING: no GPU found, fall back to CPU.', file=sys.stderr)
        gpu = [-1]
    elif max(gpu) >= count:
        raise RuntimeError('Not enough GPUs: only {} available'.format(count))
    else:
        if not override_parent and _PYTORCH_GPU_:
            # inherit from parent scope
            gpu = _PYTORCH_GPU_[-1]
    assert gpu == [-1] or min(gpu) >= 0, 'cannot mix -1 (CPU) with GPU IDs'

    # with device_ctx(gpu):
    _PYTORCH_GPU_.append(gpu)
    yield
    # restore
    _PYTORCH_GPU_.pop()

