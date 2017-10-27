"""
Notifier broadcasts message as well as the neural network parameters to the PS
"""
from surreal.comm import RedisClient
import queue
import itertools
from time import sleep
import pickle
from surreal.comm import to_str


class PSListener:
    def __init__(self, redis_client, ps_name):
        assert isinstance(redis_client, RedisClient)
        self.client = redis_client
        self.ps_name = ps_name
        self._listener_thread = None

    def run_listener_thread(self, updater):
        """
        Args:
            updater: a function that updates the policy network's parameters.
                (binary, notification_msg) -> None
        """
        # TODO: don't forget to lock PyTorch network when doing updates
        if self._listener_thread is not None:
            raise RuntimeError('Listener thread already running')

        ps_name, client = self.ps_name, self.client

        def _msg_handler(msg):
            if 'message' not in to_str(msg['type']):
                return
            msg = pickle.loads(msg['data'])
            ps_time = pickle.loads(client.get('time'))
            if msg['time'] < ps_time:
                # the parameters are newer than the message
                return
            binary = client.get(ps_name)
            updater(binary, msg['message'])

        self._listener_thread = self.client.subscribe_thread(
            ps_name, _msg_handler
        )
        return self._listener_thread

    def stop_listener_thread(self):
        self._listener_thread.stop()

