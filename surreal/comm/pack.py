"""
Defines packs that will be sent to redis
"""
import inspect
from .serializer import np_serialize, np_deserialize, binary_hash


class Pack:
    def serialize(self):
        """
        Returns:
            hask_key, binarized_data
        """
        raise NotImplementedError

    @classmethod
    def deserialize(cls, binary):
        """
        @Returns:
            the pack object created from the binary blob
        """
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError


class NumpyPack(Pack):
    unpack_init = True

    def __init__(self, data):
        self._data = data

    def serialize(self):
        binary = np_serialize(self.get_data())
        return binary_hash(binary), binary

    def get_data(self):
        return self._data

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
