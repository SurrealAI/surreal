"""
Numpy utils
"""
import numpy as np
import torch
from torch.autograd import Variable

from .gpu import GpuVariable
from surreal.utils.numpy_util import is_np_array, is_np_scalar, np_cast


def shape(x):
    if x is None:
        return None
    elif isinstance(x, np.ndarray):
        return x.shape
    else:
        return tuple(x.size())


def shapes_all(data):
    """
    Recursively walks the data (can be tuples, lists, or dict) and
    replaces a tensor with its shape tuple whenever it meets a tensor
    """
    if isinstance(data, (tuple, list)):
        ans = map(shapes_all, data)
        return type(data)(ans)
    elif isinstance(data, dict):
        return {k: shapes_all(v) for k, v in data.items()}
    elif (isinstance(data, np.ndarray)
          or torch.is_tensor(data)
          or isinstance(data, torch.autograd.Variable)
          or isinstance(data, torch.nn.Parameter)):
        return shape(data)
    else:
        return data


def print_shapes_all(data, **kwargs):
    """
    For debugging
    """
    print(shapes_all(data), **kwargs)


def dim(x):
    return len(shape(x))


def numel(x):
    return product(shape(x))


def product(L):
    return np.asscalar(np.prod(L))


def get_torch_type(x):
    if isinstance(x, list):
        return 'list'
    elif is_np_array(x) or is_np_scalar(x):
        return 'numpy'
    elif isinstance(x, Variable):
        return 'variable'
    elif torch.is_tensor(x):
        return 'tensor'
    else:
        return 'scalar'


"""
def _to_variable(x, *, gpu=-1, **flags):
    # Variable flags: requires_grad, volatile
    v = GpuVariable(x, **flags)
    return v.cuda(gpu) if gpu >= 0 else v

def _tensor_to_scalar(x):
    assert numel(x) == 1, 'tensor must have only 1 element to convert to scalar'
    return x.view(-1)[0]

def _to_numpy(x, dtype=None):
    return np.asarray(x, dtype=dtype)

def _to_tensor(x, gpu=-1):
    ans = torch.from_numpy(x)
    return ans.cuda(gpu) if gpu >=0 else ans

('numpy', 'tensor'): _to_tensor,
('numpy', 'variable'): lambda x, **kwargs: _to_variable(_to_tensor(x), **kwargs),
('numpy', 'scalar'): lambda x: np.asscalar(x),
('numpy', 'list'): lambda x: x.tolist(),
('numpy', 'numpy'): np_cast,
('tensor', 'numpy'): lambda x: (x.cpu() if x.is_cuda else x).numpy(),
('tensor', 'variable'): _to_variable,
('tensor', 'scalar'): _tensor_to_scalar,
('tensor', 'list'): ['numpy'],
('variable', 'tensor'): lambda x: x.data,
('variable', 'numpy'): ['tensor'],
('variable', 'scalar'): ['numpy'],
('variable', 'list'): ['numpy'],
('variable', 'variable'): lambda x, *, gpu=False: x.cuda() if gpu else x,
('scalar', 'tensor'): ['numpy'],
('scalar', 'variable'): 'list->variable',
('scalar', 'numpy'): 'list->numpy',
('scalar', 'list'): lambda x: [x],
('list', 'tensor'): lambda x, gpu=-1, dtype=None: \
            _to_tensor(_to_numpy(x, dtype), gpu=gpu),
('list', 'variable'): lambda x, dtype=None, **kwargs: \
            _to_variable(_to_tensor(_to_numpy(x, dtype=dtype)), **kwargs),
('list', 'scalar'): ['numpy'],
('list', 'numpy'): _to_numpy,
    
"""

def to_float_tensor(x, copy=True):
    """
    FloatTensor is the most used torch_util type, so we create a special method for it
    """
    typ = get_torch_type(x)
    if typ == 'tensor':
        assert isinstance(x, torch.FloatTensor)
        return x
    elif typ == 'variable':
        x = x.data
        assert isinstance(x, torch.FloatTensor)
        return x
    elif typ != 'numpy':
        x = np.array(x, copy=False)
    x = np_cast(x, np.float32)
    if copy:
        return torch.FloatTensor(x)
    else:
        return torch.from_numpy(x)


def to_scalar(x):
    typ = get_torch_type(x)
    if typ in ['tensor', 'variable']:
        if typ == 'variable':
            x = x.data
        assert numel(x) == 1, \
            'tensor must have only 1 element to convert to scalar'
        return x.view(-1)[0]
    elif typ == 'numpy':
        return np.asscalar(x)
    elif typ == 'list':
        assert len(x) == 1
        return x[0]
    else:
        return x

