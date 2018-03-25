"""
Shape inference methods
"""
import math
from functools import partial
from surreal.utils.pytorch import shape
from surreal.utils.common import include_keys


def is_valid_shape(shape):
    "Returns: whether a shape is valid or not"
    return all(map(lambda d : d > 0, shape))
            
            
def _expand(dim, x):
    if isinstance(x, int):
        return (x,) * dim
    else:
        assert len(x) == dim
        return x


def _expands(dim, *xs):
    "repeat vars like kernel and stride to match dim"
    return map(lambda x: _expand(dim, x), xs)


def infer_shape_convnd(dim,
                       input_shape,
                       out_channels,
                       kernel_size,
                       stride=1,
                       padding=0,
                       dilation=1,
                       has_batch=False):
    """
    http://pytorch.org/docs/nn.html#conv1d
    http://pytorch.org/docs/nn.html#conv2d
    http://pytorch.org/docs/nn.html#conv3d
    
    Args:
        dim: supports 1D to 3D
        input_shape: 
        - 1D: [channel, length]
        - 2D: [channel, height, width]
        - 3D: [channel, depth, height, width]
        has_batch: whether the first dim is batch size or not
    """
    if has_batch:
        assert len(input_shape) == dim + 2, \
            'input shape with batch should be {}-dimensional'.format(dim+2)
    else:
        assert len(input_shape) == dim + 1, \
            'input shape without batch should be {}-dimensional'.format(dim+1)
    if stride is None:
        # for pooling convention in PyTorch
        stride = kernel_size
    kernel_size, stride, padding, dilation = \
        _expands(dim, kernel_size, stride, padding, dilation)
    if has_batch:
        batch = input_shape[0]
        input_shape = input_shape[1:]
    else:
        batch = None
    _, *img = input_shape
    new_img_shape = [
        math.floor(
            (img[i] + 2 * padding[i] - dilation[i] * (kernel_size[i]- 1) - 1) // stride[i] + 1
        )
        for i in range(dim)
    ]

    return ((batch,) if has_batch else ()) + (out_channels, *new_img_shape)


def infer_shape_poolnd(dim,
                       input_shape,
                       kernel_size,
                       stride=None,
                       padding=0,
                       dilation=1,
                       has_batch=True):
    """
    The only difference from infer_shape_convnd is that `stride` default is None
    """
    if has_batch:
        out_channels = input_shape[1]
    else:
        out_channels = input_shape[0]
    return infer_shape_convnd(dim, input_shape, out_channels,
                              kernel_size, stride, padding, dilation, has_batch)


def infer_shape_transpose_convnd(dim,
                                 input_shape,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=0,
                                 output_padding=0,
                                 dilation=1,
                                 has_batch=False):
    """
    http://pytorch.org/docs/nn.html#convtranspose1d
    http://pytorch.org/docs/nn.html#convtranspose2d
    http://pytorch.org/docs/nn.html#convtranspose3d

    Args:
        dim: supports 1D to 3D
        input_shape:
        - 1D: [channel, length]
        - 2D: [channel, height, width]
        - 3D: [channel, depth, height, width]
        has_batch: whether the first dim is batch size or not
    """
    if has_batch:
        assert len(input_shape) == dim + 2, \
            'input shape with batch should be {}-dimensional'.format(dim+2)
    else:
        assert len(input_shape) == dim + 1, \
            'input shape without batch should be {}-dimensional'.format(dim+1)
    kernel_size, stride, padding, output_padding, dilation = \
        _expands(dim, kernel_size, stride, padding, output_padding, dilation)
    if has_batch:
        batch = input_shape[0]
        input_shape = input_shape[1:]
    else:
        batch = None
    _, *img = input_shape
    new_img_shape = [
        (img[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
        for i in range(dim)
    ]
    return ((batch,) if has_batch else ()) + (out_channels, *new_img_shape)



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


# TODO: deprecated
class ShapeInferenceNd(object):
    CONV_KWARG = ['stride', 'padding', 'dilation']

    def __init__(self, dim, config_list):
        """
        Args:
        dim: dimension of convolution, e.g. 2 for Conv2D
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
        self._infer_conv = partial(infer_shape_convnd, dim)
        self._infer_pool = partial(infer_shape_poolnd, dim)
        self.channels = []
        self.convs = []
        self.pools = []
        _CONV_KWARG = ['kernel'] + self.CONV_KWARG
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