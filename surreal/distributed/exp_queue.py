import sys
import weakref
import surreal.utils as U
import threading
from .zmq_struct import ZmqQueue


class ExpQueue(object):
    def __init__(self,
                 port,
                 max_size,
                 exp_handler):
        assert callable(exp_handler)
        self.max_size = max_size
        self._queue = ZmqQueue(
            port=port,
            max_size=max_size,
            start_thread=False,
            is_pyobj=True
        )
        self._dequeue_thread = None
        # ob_hash: weakref(ob)  for de-duplicating
        # when the last strong ref disappears, the ob will also be deleted here
        self._weakref_map = weakref.WeakValueDictionary()
        self._exp_handler = exp_handler

    def start_enqueue_thread(self):
        self._queue.start_enqueue_thread()

    def _dequeue_loop(self):  # blocking
        while True:
            exp_list, ob_storage = self._queue.get()
            assert isinstance(exp_list, list)
            assert isinstance(ob_storage, dict)
            for exp in exp_list:
                exp_deflated = self._retrieve_storage(exp, ob_storage)
                self._exp_handler(exp_deflated)
            # clean up ref counts
            del exp_list, ob_storage, exp, exp_deflated

    def start_dequeue_thread(self):
        """
        handler function takes an experience tuple
        ([obs], action, reward, done, info)
        inserts it into a priority replay data structure.
        """
        if self._dequeue_thread is not None:
            raise ValueError('Dequeue thread is already running')
        self._dequeue_thread = U.start_thread(self._dequeue_loop)
        return self._dequeue_thread

    def _retrieve_storage(self, exp, storage):
        """
        Args:
            exp: a nested dict or list
                Only dict keys that end with `_hash` will be retrieved.
                The processed key will see `_hash` removed
            storage: chunk of storage sent with the exps
        """
        if isinstance(exp, list):
            for i, e in enumerate(exp):
                exp[i] = self._retrieve_storage(e, storage)
        elif isinstance(exp, dict):
            for key in list(exp.keys()):  # copy keys
                if key.endswith('_hash'):
                    new_key = key[:-len('_hash')]  # delete suffix
                    exp[new_key] = self._retrieve_storage(exp[key], storage)
                    del exp[key]
        elif isinstance(exp, str):
            exphash = exp
            if exphash in self._weakref_map:
                return self._weakref_map[exphash]
            else:
                self._weakref_map[exphash] = storage[exphash]
                return storage[exphash]
        return exp

    def size(self):
        return len(self._queue)

    __len__ = size

    def occupancy(self):
        "ratio of current size / max size"
        return 1. * self.size() / self.max_size

    def weakref_keys(self):
        return list(self._weakref_map.keys())

    def weakref_size(self):
        return len(self._weakref_map)

    def weakref_counts(self):
        return {key: sys.getrefcount(value) - 3  # counting itself incrs ref
                for key, value in self._weakref_map.items()}
