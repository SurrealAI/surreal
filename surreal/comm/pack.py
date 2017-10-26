"""
Defines packs that will be sent to redis
"""
import inspect
from .serializer import np_serialize, np_deserialize, binary_hash


class Pack:
    def serialize(self):
        raise NotImplementedError

    @classmethod
    def deserialize(cls, binary):
        """
        @Returns:
            the pack object created from the binary blob
        """
        raise NotImplementedError

    def get_key(self):
        """
        Key of this pack to be stored in Redis. None if this pack is not stored.
        """
        return None

    def get_data(self):
        raise NotImplementedError

    def jsonify(self):
        # TODO
        pass


class NumpyPack(Pack):
    unpack_init = True

    def __init__(self, data):
        self._data = data
        self._cache = None

    def serialize(self):
        self._cache = np_serialize(self.get_data())
        return self._cache

    def get_data(self):
        return self._data

    def get_key(self):
        binary = self._cache if self._cache else self.serialize()
        return binary_hash(binary)

    def __getattr__(self, key):
        if not isinstance(self._data, dict) or key in dir(self):
            return object.__getattribute__(self, key)
        else:
            return self._data[key]

    @classmethod
    def deserialize(cls, binary):
        data = np_deserialize(binary)
        if cls.unpack_init:
            return cls(**data)
        else:
            return cls(data)
