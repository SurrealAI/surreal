from .pack import Pack
from .serializer import np_serialize, np_deserialize, binary_hash


class ObsPack(Pack):
    def __init__(self, obs):
        """
        Args:
            obs: numpy (or other heavy data structure) of the observation
        """
        self.obs = obs
        self._cache = None

    def serialize(self):
        self._cache = np_serialize(self.obs)
        return self._cache

    def get_data(self):
        return self.obs

    def get_key(self):
        binary = self._cache if self._cache else self.serialize()
        return binary_hash(binary)

    @classmethod
    def deserialize(cls, binary):
        return cls(np_deserialize(binary))


class ExpPack(Pack):
    def __init__(self, obs_pointers, action, reward, info):
        assert isinstance(obs_pointers, list)
        assert isinstance(reward, (float, int))
        assert isinstance(info, dict)
        self.obs_pointers = obs_pointers
        self.obses = None # not serialized
        self.action = action
        self.reward = reward
        self.info = info
        self._cache = None

    def serialize(self):
        self._cache = np_serialize(self.get_data())
        return self._cache

    def get_data(self):
        return {
            'obs_pointers': self.obs_pointers,
            'action': self.action,
            'reward': self.reward,
            'info': self.info
        }

    def get_key(self):
        binary = self._cache if self._cache else self.serialize()
        return binary_hash(binary)

    @classmethod
    def deserialize(cls, binary):
        return cls(**np_deserialize(binary))
