"""
Agent side.
Send experience chunks (buffered) to Replay node.
"""
import surreal.utils as U
from surreal.session import PeriodicTracker
from .zmq_struct import ZmqPushClient


class ExpBuffer(object):
    def __init__(self):
        # ([obs], reward, done, info)
        self.exp_tuples = []
        # {obs_hash: [obs, ref_count]}
        self.ob_storage = {}

    def add(self, obs, *other_info):
        U.assert_type(obs, list)
        ob_hashes = []
        for ob in obs:
            hsh = U.pyobj_hash(ob)
            ob_hashes.append(hsh)
            if hsh not in self.ob_storage:
                self.ob_storage[hsh] = ob
        self.exp_tuples.append([ob_hashes, *other_info])

    def flush(self):
        binary = U.serialize((self.exp_tuples, self.ob_storage))
        # U.print_('SIZE', len(binary), 'Exp', self.exp_tuples, 'ob', self.ob_storage)
        self.exp_tuples = []
        self.ob_storage = {}
        return binary


class ExpSender(object):
    def __init__(self, *,
                 host,
                 port,
                 # TODO add flush_time
                 flush_iteration):
        """
        Args:
            flush_iteration: how many send() calls before we flush the buffer
        """
        U.assert_type(flush_iteration, int)
        self._client = ZmqPushClient(
            host=host,
            port=port,
            is_pyobj=False,
        )
        self._exp_buffer = ExpBuffer()
        self._flush_tracker = PeriodicTracker(flush_iteration)

    def send(self, obs, action, reward, done, info):
        """
        """
        U.assert_type(obs, list)
        self._exp_buffer.add(obs, action, reward, done, info)
        if self._flush_tracker.track_increment():
            exp_binary = self._exp_buffer.flush()
            self._client.push(exp_binary)
            return U.binary_hash(exp_binary)
        else:
            return None
