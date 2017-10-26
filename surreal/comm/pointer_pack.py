from .pack import NumpyPack
from .serializer import np_serialize, np_deserialize


class PointerPack(NumpyPack):
    unpack_init = True

    def __init__(self, obs_pointers, exp_pointer, replay_info):
        """
        Args:
            obs_pointers: list of observation hashes
            exp_pointer: experience tuple hash
            replay_info: TD error or other information used in prioritization
        """
        assert isinstance(obs_pointers, list)
        assert isinstance(exp_pointer, str)
        assert isinstance(replay_info, dict)
        super().__init__({
            'obs_pointers': obs_pointers,
            'exp_pointer': exp_pointer,
            'replay_info': replay_info,
        })

