#from .module import Module
from .conv import conv_fc_init
import torch.nn as nn
import torch.nn.functional as F
import torchx as tx
import torchx.nn as nnx
from surreal.utils.common import iter_last


def fc_layers(input_size, output_size, hiddens, initializer='xavier'):
    assert isinstance(hiddens, (list, tuple))
    fcs = nn.ModuleList() # IMPORTANT for .cuda() to work!!
    layers = [input_size] + hiddens + [output_size]
    for prev, next in zip(layers[:-1], layers[1:]):
        fcs.append(nn.Linear(prev, next))
    if initializer == 'xavier':
        conv_fc_init(fcs)
    return fcs


class MLP(nnx.Module):
    def __init__(self, input_size, output_size, hiddens, activation=None):
        super().__init__()
        if activation is None:
            self.activation = F.relu
        else:
            raise NotImplementedError # TODO: other activators
        self.layers = fc_layers(input_size=input_size,
                                output_size=output_size,
                                hiddens=hiddens)

    def reinitialize(self):
        conv_fc_init(self.layers)

    def forward(self, x):
        for is_last, fc in iter_last(self.layers):
            x = fc(x)
            if not is_last:
                x = self.activation(x)
        return x

