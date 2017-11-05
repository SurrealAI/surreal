"""
Defines packs that will be sent to Redis
"""
import inspect
import pickle
import surreal.utils as U


class Pack(object):
    def __init__(self, data, serializer=None):
        """
        Args:
            data
            serializer: defaults to pickle.dumps
        """
        self._data = data
        if serializer is None:
            self._serializer = pickle.dumps
        else:
            self._serializer = serializer

    def serialize(self):
        """
        Returns:
            hask_key, binarized_data
        """
        binary = self._serializer(self._data)
        return U.binary_hash(binary), binary

    @staticmethod
    def deserialize(binary, deserializer=None):
        """
        Returns:
            deserialized data
        """
        if deserializer is None:
            deserializer = pickle.loads
        return deserializer(binary)

    @property
    def data(self):
        return self._data


class ExpPack(Pack):
    def serialize(self):
        """
        Also insert exp_pointer (i.e. hash of this exp dict before serialize)
        """
        pointer, binary = super().serialize()
        dat = self._data.copy()
        dat['exp_pointer'] = pointer
        return pointer, self._serializer(dat)

    def get_key(self):
        return 'exp:' + super().get_key()


class ExpPointerPack(ExpPack):
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


class ExpFullPack(ExpPack):
    def __init__(self, obses, action, reward, done, info):
        assert isinstance(obses, list)
        assert isinstance(reward, (float, int))
        assert isinstance(info, dict)
        super().__init__({
            'obses': obses,
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        })


class ObsPack(Pack):
    @staticmethod
    def deserialize(binary):
        return pickle.loads(binary)

    def get_key(self):
        return 'obs:' + super().get_key()
