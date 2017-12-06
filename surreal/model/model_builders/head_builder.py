import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadBuilder(nn.Module):

    def __init__(self,
                 output_spec):

        super(HeadBuilder, self).__init__()
        self._dims = output_spec.dim
        self._type = output_spec.type
        self._head = None

    def forward(self, obs):

        in_dim = obs.size(1)
        out_dim = self._dims[0]

        if self._head is None:

            if self._type == 'discrete':
                self._head = nn.Linear(in_dim, out_dim)

            elif self._type == 'continuous':
                self._head = nn.Linear(in_dim, out_dim)

            elif self._type == 'gaussian':
                self._head = {
                    'mean': nn.Linear(in_dim, out_dim),
                    'std': nn.Linear(in_dim, out_dim)
                }

            elif self._type == 'scalar':
                self._head = nn.Linear(in_dim, 1)

            elif self._type == 'distributional':
                raise NotImplementedError

            else:
                raise ValueError('Unknown head type: {}'.format(type))

        if isinstance(self._head, dict):
            out = dict()
            for k, m in self._head.items():
                out[k] = m(obs)
        else:
            out = self._head(obs)

        return out

