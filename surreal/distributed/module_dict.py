import numpy as np
import torch
from surreal.utils.numpy_util import np_cast
import surreal.utils as U
import torchx.nn as nnx


class ModuleDict(object):
    """
    Two-step serialization.
    1. Each element's state_dict() is called
    2. The overall dict is then pickled.
    """
    def __init__(self, module_dict):
        U.assert_type(module_dict, dict)
        for k, m in module_dict.items():
            U.assert_type(k, str, 'Key "{}" must be string.'.format(k))
            U.assert_type(m, nnx.Module,
                          '"{}" must be torchx.nn.Module.'.format(m))
        self._module_dict = module_dict

    def dumps(self):
        """
            Dump content into binary

        Returns:
            bytes
        """
        bin_dict = {}
        for k, m in self._module_dict.items():
            state_dict = m.state_dict()
            for key in state_dict:
                state_dict[key] = state_dict[key].cpu().numpy()
            bin_dict[k] = state_dict
        return U.serialize(bin_dict)

    def loads(self, binary):
        """
            Load from binary ()

        Args:
            binary: output of ModuleDict.dumps
        """
        numpy_dict = U.deserialize(binary)
        self.load(numpy_dict)

    def load(self, numpy_dict):
        """
            Load data from a numpy_dictionary

        Args:
            numpy_dict: {
                <module_dict_key>: {
                    <paramter_key_module_dict_entries>: np.array
                }
            }
        """
        for key in numpy_dict:
            for k in numpy_dict[key]:
                numpy_dict[key][k] = torch.from_numpy(
                    np_cast(numpy_dict[key][k], np.float32))
        for k, m in self._module_dict.items():
            m.load_state_dict(numpy_dict[k])
