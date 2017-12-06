import torch
import torch.nn as nn
import torch.nn.functional as F
import surreal.utils as U


class TorsoBuilder(nn.Module):

    def __init__(self,
                 input_spec,
                 conv_spec=None,
                 mlp_spec=None,
                 rnn_spec=None):

        super(TorsoBuilder, self).__init__()

        if conv_spec:
            assert len(input_spec['dims']) == 3, 'torso requires 3-dimensional inputs to conv layers.'
        elif mlp_spec:
            assert len(input_spec['dims']) == 1, 'torso requires flat inputs to fc layers.'
        elif rnn_spec:
            raise NotImplementedError
        else:
            raise ValueError('torso must contain at least one layer.')

        self._conv_spec = conv_spec
        self._mlp_spec = mlp_spec
        self._rnn_spec = rnn_spec

        self._input_spec = input_spec
        self.conv_module = None
        self.mlp_module = None
        self.rnn_module = None

    def forward(self, obs):

        # verify obs shapes
        for k, d in enumerate(self._input_spec['dims']):
            assert obs.size(k+1) == d

        out = obs

        if self._conv_spec:
            if not self.conv_module:
                self.conv_module = build_conv_module(self._input_spec,
                                                     **self._conv_spec)
            out = self.conv_module(out)

        if self._mlp_spec:
            # lazy evaluation for shape inference
            out = out.view(out.size(0), -1)
            if not self.mlp_module:
                mlp_input_spec = {'dims': [out.size(1)]}
                self.mlp_module = build_mlp_module(mlp_input_spec,
                                                   **self._mlp_spec)
            out = self.mlp_module(out)

        return out


def build_conv_module(input_spec,
                      out_channels,
                      kernel_sizes,
                      strides=None,
                      paddings=None,
                      dilations=None,
                      use_batch_norm=False):

    layers = []
    in_channels = input_spec['dims'][0]

    for i in range(len(kernel_sizes)):

        stride = strides[i] if strides else 1
        padding = paddings[i] if paddings else 0
        dilation = dilations[i] if dilations else 1

        conv2d = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels[i],
                           kernel_size=kernel_sizes[i],
                           stride=stride,
                           padding=padding,
                           dilation=dilation)

        in_channels = out_channels[i]

        if use_batch_norm:
            layers += [conv2d, nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]

    layers = nn.Sequential(*layers)
    return layers


def build_mlp_module(input_spec,
                     sizes,
                     use_dropout=False):

    assert len(input_spec['dims']) == 1, 'MLP requires flat inputs.'
    assert len(sizes) > 0, 'MLP must have at least one fc layer.'

    layers = []
    prev_dim = input_spec['dims'][0]
    for i in range(len(sizes)):
        layers += [nn.Linear(prev_dim , sizes[i]), nn.ReLU(inplace=True)]
        if use_dropout:
            layers += [nn.Dropout()]
        prev_dim = sizes[i]

    layers = nn.Sequential(*layers)
    return layers
