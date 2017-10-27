from .pack import NumpyPack
from .serializer import np_serialize, np_deserialize, binary_hash


class ExpPack(NumpyPack):
    unpack_init = True

    def __init__(self, obs_pointers, action, reward, done, info):
        assert isinstance(obs_pointers, list)
        assert isinstance(reward, (float, int))
        assert isinstance(info, dict)
        super().__init__({
            'obs_pointers': obs_pointers,
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        })


class ObsPack(NumpyPack):
    unpack_init = False

    @property
    def obs(self):
        return self._data

