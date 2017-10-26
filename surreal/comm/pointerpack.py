from .pack import Pack
from .serializer import np_serialize, np_deserialize


class PointerPack(Pack):
    def __init__(self, obs_pointers, exp_pointer, delta):
        """
        Args:
            obs_pointers: list of observation hashes
            exp_pointer: experience tuple hash
            delta: TD error or other information used in prioritized replay
        """
        assert isinstance(obs_pointers, list)
        assert isinstance(exp_pointer, str)
        self.obs_pointers = obs_pointers
        self.exp_pointer = exp_pointer
        self.delta = delta

    def serialize(self):
        return np_serialize(self.get_data())

    def get_data(self):
        return {
            'obs_pointers': self.obs_pointers,
            'exp_pointer': self.exp_pointer,
            'delta': self.delta,
        }

    @classmethod
    def deserialize(cls, binary):
        return cls(**np_deserialize(binary))
