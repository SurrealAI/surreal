"""
Defines packs that will be sent to redis
"""
import inspect
import pickle
import surreal.utils as U


class Pack(object):
    def __init__(self, data, serializer=None):
        """
        Args:
            serializer defaults to pickle.dumps
        """
        self._data = data
        if serializer is None:
            self._serializer = pickle.dumps
        else:
            self._serializer = serializer
        self._cache = None

    def serialize(self):
        """
        Returns:
            hask_key, binarized_data
        """
        if self._cache is None:
            self._cache = self._serializer(self._data)
        return self._cache

    def get_key(self):
        raise NotImplementedError

    @classmethod
    def deserialize(cls, binary):
        """
        @Returns:
            the pack object created from the binary blob
        """
        raise NotImplementedError

    @property
    def data(self):
        return self._data


class NumpyPack(Pack):
    unpack_init = True

    def __init__(self, data):
        super().__init__(data, serializer=U.np_serialize)

    def get_key(self):
        return U.binary_hash(self.serialize())

    def __getattr__(self, key):
        if not isinstance(self._data, dict) or key in dir(self):
            return object.__getattribute__(self, key)
        else:
            return self._data[key]

    @classmethod
    def deserialize(cls, binary):
        data = U.np_deserialize(binary)
        if cls.unpack_init:
            return cls(**data)
        else:
            return cls(data)
