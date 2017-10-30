"""
Numpy and math operations
"""
import numpy as np


def min_at(values):
    "Returns: (min, min_i)"
    if not values:
        return None, None
    return min( (v, i) for i, v in enumerate(values) )


def max_at(values):
    "Returns: (max, max_i)"
    if not values:
        return None, None
    return max( (v, i) for i, v in enumerate(values) )


def sum_pow(p, n_begin, n_end):
    """
    Power summation, N inclusive.
    \sum_{n=N_begin}^{N_end} p^n
    """
    return (p**(n_end+1) - p**n_begin) / (p - 1.0)


def ceildiv(a, b):
    """
    Ceiling division, equivalent to math.ceil(1.0*a/b) but much faster.
    ceildiv(19, 7) == 3
    ceildiv(21, 7) == 3
    ceildiv(22, 7) == 4
    """
    return - (-a // b)


def is_div(a, b):
    " Returns: bool - does `a` divide `b`. "
    return int(a) % int(b) == 0


def cum_sum(seq):
    """
    Cumulative sum (include 0)
    """
    s = 0
    cumult = [0]
    for n in seq:
        s += n
        cumult.append(s)
    return cumult
#     return [sum(seq[:i]) for i in range(len(seq) + 1)]


def is_np_array(L, dtype=None):
    if dtype is None:
        return isinstance(L, np.ndarray)
    return isinstance(L, np.ndarray) and np.issubdtype(L.dtype, dtype)


def is_float_array(L):
    return is_np_array(L, np.float)


def is_int_array(L):
    return is_np_array(L, np.int_)


def is_np_scalar(x):
    """
    Check np types like np.int64
    """
    return isinstance(x, np.generic)


def is_np_int(x):
    return isinstance(x, np.int_)


def is_np_float(x):
    return isinstance(x, np.float_)


def np_cast(x, dtype):
    if dtype is None or x.dtype == dtype:
        return x
    else:
        return x.astype(dtype)


def compare(a, b, *, tol=1e-6):
    """
    if ||a - b|| < tol, return 0
    otherwise return float(a > b)
    """
    if abs(a - b) < tol:
        return 0.0
    elif a > b:
        return 1.0
    else:
        return -1.0


def np_clip_(x, min=None, max=None):
    """
    In-place clip. Also supports indexing
    """
    return np.clip(x, min, max, out=x)
