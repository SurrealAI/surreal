import queue
import itertools
import time
import surreal.utils as U
import threading
from .redis_client import RedisClient
from .obs_fetcher import ObsFetcher
from .obs_ref_count import decr_ref_count


class _EnqueueThread(U.StoppableThread):
    def __init__(self,
                 client,
                 queue,
                 sampler,
                 fetcher):
        self._client = client
        self._queue = queue
        self._sampler = sampler
        self._fetcher = fetcher
        super().__init__()

    def run(self):
        for i in itertools.count():
            if self.is_stopped():
                break
            while True:
                exp_dicts = self._sampler()
                if exp_dicts is None:  # start_sample_condition not met
                    time.sleep(.5)
                else:
                    exps = self._fetcher.fetch(exp_dicts)
                    # decr ref counts which were incr'ed by sample()
                    # evict the exps which should have been evicted by insert()
                    obs_pointers = []
                    for exp in exps:
                        if 'obs_pointers' in exp:
                            obs_pointers.extend(exp['obs_pointers'])
                    decr_ref_count(self._client, obs_pointers, delete=True)
                    # block if the queue is full
                    self._queue.put(exps, block=True, timeout=None)
                    break


class BatchFetchQueue(object):
    def __init__(self, redis_client, maxsize):
        self._queue = queue.Queue(maxsize=maxsize)
        assert isinstance(redis_client, RedisClient)
        self._client = redis_client
        self._fetcher = ObsFetcher(redis_client)
        self._enqueue_thread = None

    def start_enqueue_thread(self, sampler):
        """
        Producer thread, runs sampler function on a priority replay structure
        Args:
            sampler: function batch_i -> list
                returns exp_dicts with 'obs_pointers' field
            start_sample_condition: function () -> bool
                begins sampling only when this returns True.
                Example: when the replay memory exceeds a threshold size
            start_sample_condvar: threading.Condition()
                notified by Replay.insert() when start sampling condition is met
            evict_lock: do not evict in the middle of fetching exp, otherwise
                we might fetch a null exp that just got evicted.
                locked by Replay.evict()
        """
        if self._enqueue_thread is not None:
            raise RuntimeError('Enqueue thread is already running')
        self._enqueue_thread = _EnqueueThread(
            client=self._client,
            queue=self._queue,
            sampler=sampler,
            fetcher=self._fetcher,
        )
        self._enqueue_thread.start()
        return self._enqueue_thread

    def stop_enqueue_thread(self):
        self._enqueue_thread.stop()
        self._enqueue_thread = None

    def dequeue(self):
        """
        Called by the neural network, draw the next batch of experiences
        """
        return self._queue.get(block=True, timeout=None)

    def queue_size(self):
        return self._queue.qsize()