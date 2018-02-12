import pickle
import surreal.utils as U


class ModuleDict(object):
    """
    Two-step serialization.
    1. Each element's state_dict() is called
    2. The overall dict is then pickled.
    """
    def __init__(self, module_dict):
        U.assert_type(module_dict, dict)
        for k, m in module_dict.items():
            U.assert_type(k, str), 'Key "{}" must be string.'.format(k)
            U.assert_type(m, U.Module), \
            '"{}" must be surreal.utils.pytorch.Module.'.format(m)
        self._module_dict = module_dict

    def dumps(self):
        bin_dict = {}
        for k,m in self._module_dict.items():
            state_dict = m.state_dict()
            for key in state_dict:
                state_dict[key] = state_dict[key].cpu()
            bin_dict[k] = state_dict
        return U.serialize(bin_dict)

    def loads(self, binary):
        bin_dict = U.deserialize(binary)
        for k, m in self._module_dict.items():
            m.load_state_dict(bin_dict[k])

