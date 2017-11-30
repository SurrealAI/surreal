"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
import queue
import itertools
from time import sleep
import pickle
import surreal.utils as U


class Listener(object):
    def __init__(self, redis_client, name='ps'):
        U.assert_type(redis_client, RedisClient)
        self._client = redis_client
        self._name = name
        self._listener_thread = None

    def update(self, binary, message=''):
        """
        Abstract method that updates the parameters on the agent side.

        Args:
            binary: binarized parameters
            message: associated meta-data, if any. Can be empty.
        """
        raise NotImplementedError

    def run_listener_thread(self):
        if self._listener_thread is not None:
            raise RuntimeError('Listener thread already running')

        ps_name, client = self._name, self._client

        def _msg_handler(msg):
            if 'message' not in U.bytes2str(msg['type']):
                return
            msg = pickle.loads(msg['data'])
            ps_time = pickle.loads(client.get('time'))
            if msg['time'] < ps_time:
                # the parameters are newer than the message
                return
            binary = client.get(ps_name)
            self.update(binary, msg['message'])

        self._listener_thread = self._client.start_subscribe_thread(
            ps_name, _msg_handler
        )
        return self._listener_thread

    def stop_listener_thread(self):
        self._listener_thread.stop()

