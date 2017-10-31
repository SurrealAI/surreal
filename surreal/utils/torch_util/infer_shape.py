"""
Shape inference methods
"""
import math
from functools import partial
from surreal.utils.torch_util import shape
from surreal.utils.common import include_keys


CONV_KWARG = ['stride', 'padding', 'dilation']


def is_valid_shape(shape):
    "Returns: whether a shape is valid or not"
    return all(map(lambda d : d > 0, shape))
            
            
def _expand(D, x):
    if isinstance(x, int):
        return (x,) * D
    else:
        assert len(x) == D
        return x


def _expands(D, *xs):
    "repeat vars like kernel and stride to match dim"
    return map(lambda x: _expand(D, x), xs)


def infer_shape_convnd(D, input_shape, out_channels, 
                       kernel, stride=1, padding=0, dilation=1):
    """
    http://pytorch.org/docs/nn.html#conv1d
    http://pytorch.org/docs/nn.html#conv2d
    http://pytorch.org/docs/nn.html#conv3d
    """
    assert len(input_shape) in [D+1, D+2], \
        'expect input_shape to have {} (batchless) or {} (with batch) dims'.format(D+1, D+2)
    if stride is None:
        # for pooling convention in PyTorch
        stride = kernel
    kernel, stride, padding, dilation = _expands(D, kernel, stride, padding, dilation)
    has_batch = len(input_shape)==D+2
    if has_batch:
        batch = input_shape[0]
        input_shape = input_shape[1:]
    _, *img = input_shape
    new_img = [] # a single image's img
    for i in range(D):
        n = math.floor((img[i] + 2*padding[i] - dilation[i]*(kernel[i]-1) -1)//stride[i] + 1)
        new_img.append(n)

    return ((batch,) if has_batch else ()) + (out_channels, *new_img)


def infer_shape_poolnd(D, input_shape, out_channels, 
                       kernel, stride=None, padding=0, dilation=1):
    """
    The only difference from infer_shape_convnd is that `stride` default is None
    """
    return infer_shape_convnd(D, input_shape, out_channels,
                              kernel, stride, padding, dilation)


def infer_shape_deconvnd():
    """
    TODO: deconvolution layers
    http://pytorch.org/docs/nn.html#convtranspose1d
    http://pytorch.org/docs/nn.html#convtranspose2d
    http://pytorch.org/docs/nn.html#convtranspose3d
    """
    pass


infer_shape_conv1d = partial(infer_shape_convnd, 1)
infer_shape_conv2d = partial(infer_shape_convnd, 2)
infer_shape_conv3d = partial(infer_shape_convnd, 3)


infer_shape_maxpool1d = partial(infer_shape_poolnd, 1)
infer_shape_maxpool2d = partial(infer_shape_poolnd, 2)
infer_shape_maxpool3d = partial(infer_shape_poolnd, 3)


"""
http://pytorch.org/docs/nn.html#avgpool1d
http://pytorch.org/docs/nn.html#avgpool2d
http://pytorch.org/docs/nn.html#avgpool3d
"""
infer_shape_avgpool1d = partial(infer_shape_maxpool1d, dilation=1)
infer_shape_avgpool2d = partial(infer_shape_maxpool2d, dilation=1)
infer_shape_avgpool3d = partial(infer_shape_maxpool3d, dilation=1)


class ShapeInferenceNd(object):
    def __init__(self, D, config_list):
        """
        Args:
        D: dimension of convolution, e.g. 2 for Conv2D
        config_list:
          Architecture configs for each conv2d and pooling. 
          Each entry in the list is a dict:
          - channel: output channel size
          - kernel
          - stride
          - padding
          - dilation
          - pool (optional): a sub-dict that configures a pooling layer
              - kernel
              - stride
              - padding
              - dilation: applies only to MaxPool
        """
        self._infer_conv = partial(infer_shape_convnd, D)
        self._infer_pool = partial(infer_shape_poolnd, D)
        self.channels = []
        self.convs = []
        self.pools = []
        _CONV_KWARG = ['kernel'] + CONV_KWARG
        for config in config_list:
            pool = config.get('pool', None)
            if pool is not None:
                pool = include_keys(_CONV_KWARG, pool)
            self.pools.append(pool)
            self.channels.append(config['channel'])
            conv = include_keys(_CONV_KWARG, config)
            self.convs.append(conv)
        
    
    def __call__(self, x):
        """
        Args:
          x: can be either a shape tuple or an actual tensor
        
        Returns:
          inferred shape
        """
        if not isinstance(x, (tuple, list)):
            x = shape(x)
        for channel, conv, pool in zip(self.channels, self.convs, self.pools):
            x = self._infer_conv(x, channel, **conv)
            if pool:
                x = self._infer_pool(x, channel, **pool)
        return x
    

ShapeInference1D = partial(ShapeInferenceNd, 1)
ShapeInference2D = partial(ShapeInferenceNd, 2)
ShapeInference3D = partial(ShapeInferenceNd, 3)