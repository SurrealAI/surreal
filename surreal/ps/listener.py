"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
from surreal.comm import RedisClient
import queue
import itertools
from time import sleep
import pickle
from surreal.utils import assert_type, bytes2str


class Listener(object):
    def __init__(self, redis_client, name='ps'):
        assert_type(redis_client, RedisClient)
        self._client = redis_client
        self._name = name
        self._listener_thread = None

    def update(self, binary, message):
        """
        Updates the policy network's parameters.
        """
        raise NotImplementedError

    def run_listener_thread(self):
        # TODO: don't forget to lock PyTorch network when doing updates
        if self._listener_thread is not None:
            raise RuntimeError('Listener thread already running')

        ps_name, client = self._name, self._client

        def _msg_handler(msg):
            if 'message' not in bytes2str(msg['type']):
                return
            msg = pickle.loads(msg['data'])
            ps_time = pickle.loads(client.get('time'))
            if msg['time'] < ps_time:
                # the parameters are newer than the message
                return
            binary = client.get(ps_name)
            self.update(binary, msg['message'])

        self._listener_thread = self._client.subscribe_thread(
            ps_name, _msg_handler
        )
        return self._listener_thread

    def stop_listener_thread(self):
        self._listener_thread.stop()

