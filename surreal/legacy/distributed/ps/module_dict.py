import pickle
import surreal.utils as U


class ModuleDict(object):
    """
    Two-step serialization.
    1. All objects of surreal.utils.pytorch.Module will be serialized by a
        specialized method that flattens the tensors and compact into a binary.
    2. The outside dict is then pickled.
    """
    def __init__(self, module_dict):
        U.assert_type(module_dict, dict)
        for k, m in module_dict.items():
            U.assert_type(k, str), 'Key "{}" must be string.'.format(k)
            U.assert_type(m, U.Module), \
            '"{}" must be surreal.utils.pytorch.Module.'.format(m)
        self._module_dict = module_dict

    def dumps(self):
        bin_dict = {k: m.parameters_to_binary()
                    for k, m in self._module_dict.items()}
        return pickle.dumps(bin_dict)

    def loads(self, binary):
        bin_dict = pickle.loads(binary)
        for k, m in self._module_dict.items():
            m.parameters_from_binary(bin_dict[k])

