from .pack import NumpyPack
from .serializer import np_serialize, np_deserialize


class PointerPack(NumpyPack):
    unpack_init = True

    def __init__(self, obs_pointers, exp_pointer, exp_without_obs):
        """
        Args:
            obs_pointers: list of observation hashes
            exp_pointer: experience tuple hash
            delta: TD error or other information used in prioritized replay
        """
        assert isinstance(obs_pointers, list)
        assert isinstance(exp_pointer, str)
        super().__init__({
            'obs_pointers': self.obs_pointers,
            'exp_pointer': self.exp_pointer,
            'delta': self.delta,
        })

