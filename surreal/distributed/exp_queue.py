import sys
import weakref
import surreal.utils as U
from .zmq_struct import ZmqQueue


class _DequeueThread(U.StoppableThread):
    def __init__(self, queue, weakref_map, exp_handler):
        self._queue = queue
        self._weakref_map = weakref_map
        self._exp_handler = exp_handler
        super().__init__()

    def run(self):
        while True:
            if self.is_stopped():
                break
            exp_tuples, ob_storage = self._queue.get()
            exp_tuple, ob_list = None, None
            for exp_tuple in exp_tuples:
                # deflate exp_tuple
                ob_list = exp_tuple[0]
                U.assert_type(ob_list, list)
                for i, ob_hash in enumerate(ob_list):
                    if ob_hash in self._weakref_map:
                        ob_list[i] = self._weakref_map[ob_hash]
                    else:
                        ob_list[i] = ob_storage[ob_hash]
                        self._weakref_map[ob_hash] = ob_list[i]
                self._exp_handler(exp_tuple)
            # clean up ref counts
            del exp_tuples, ob_storage, exp_tuple, ob_list


class ExpQueue(object):
    def __init__(self,
                 port,
                 max_size,
                 exp_tuple_handler):
        assert callable(exp_tuple_handler)
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
        self._exp_tuple_handler = exp_tuple_handler

    def start_enqueue_thread(self):
        self._queue.start_enqueue_thread()

    def start_dequeue_thread(self):
        """
        handler function takes an experience tuple
        ([obs], action, reward, done, info)
        inserts it into a priority replay data structure.
        """
        if self._dequeue_thread is not None:
            raise ValueError('Dequeue thread is already running')
        self._dequeue_thread = _DequeueThread(
            self._queue,
            self._weakref_map,
            self._exp_tuple_handler,
        )
        self._dequeue_thread.daemon = True
        self._dequeue_thread.start()
        return self._dequeue_thread

    def stop_dequeue_thread(self):
        self._dequeue_thread.stop()
        self._dequeue_thread = None

    def size(self):
        return len(self._queue)

    __len__ = size

    def weakref_keys(self):
        return list(self._weakref_map.keys())

    def weakref_size(self):
        return len(self._weakref_map)

    def weakref_counts(self):
        return {key: sys.getrefcount(value) - 3  # counting itself incrs ref
                for key, value in self._weakref_map.items()}
