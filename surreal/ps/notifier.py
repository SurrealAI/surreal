"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
import time
import pickle
from surreal.comm import RedisClient


class PSNotifier(object):
    def __init__(self, redis_client, ps_name):
        assert isinstance(redis_client, RedisClient)
        self.client = redis_client
        self.ps_name = ps_name

    def update(self, binary, message):
        """
        Also include a monotically increasing timing info to discard
        outdated message on the listener side
        """
        time_info = time.perf_counter()
        msg = {
            'message': message,
            'time': time_info
        }
        self.client.set(self.ps_name, binary)
        self.client.set('time', pickle.dumps(time_info))
        self.client.publish(self.ps_name, pickle.dumps(msg))
