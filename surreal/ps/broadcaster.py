"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
import time
import pickle
from surreal.distributed import RedisClient
import surreal.utils as U


class Broadcaster(object):
    def __init__(self, redis_client, name='ps'):
        """
        Args:
        """
        U.assert_type(redis_client, RedisClient)
        self._client = redis_client
        self._name = name

    def broadcast(self, binary, message):
        """
        Also include a monotically increasing timing info to discard
        outdated message on the listener side

        Args:
            binary: preprocess the network in subclasses into binary
            message: strings or any JSONable data
        """
        time_info = time.perf_counter()
        msg = {
            'message': message,
            'time': time_info
        }
        self._client.set(self._name, binary)
        self._client.set('time', pickle.dumps(time_info))
        self._client.publish(self._name, pickle.dumps(msg))
