import queue
import itertools
from time import sleep
from surreal.distributed import RedisClient, ExpPack
from surreal.utils.common import StoppableThread
from .obs_fetcher import ObsFetcher


class _EnqueueThread(StoppableThread):
    def __init__(self, queue, sampler, fetcher, start_sample_condition):
        """
        start_sample_condition(): begins sampling only when this returns True.
            Example: when the replay memory exceeds a threshold size
        """
        self._queue = queue
        self._sampler = sampler
        self._fetcher = fetcher
        self._start_sample_condition = start_sample_condition
        super().__init__()

    def run(self):
        while not self._start_sample_condition():
            if self.is_stopped():
                break
            sleep(0.5)
        for i in itertools.count():
            if self.is_stopped():
                break
            exp_dicts = self._sampler(i)
            exps = self._fetcher.fetch(exp_dicts)
            # block if the queue is full
            self._queue.put(exps, block=True, timeout=None)


class ExpFetcherQueue:
    def __init__(self, redis_client, maxsize):
        self._queue = queue.Queue(maxsize=maxsize)
        assert isinstance(redis_client, RedisClient)
        self._fetcher = ObsFetcher(redis_client)
        self._enqueue_thread = None

    def start_enqueue_thread(self, sampler, start_sample_condition):
        """
        Producer thread, runs sampler function on a priority replay structure
        Args:
            sampler: batch_i -> list of exp_dicts with 'obs_pointers' field
        """
        if self._enqueue_thread is not None:
            raise RuntimeError('Enqueue thread is already running')
        self._enqueue_thread = _EnqueueThread(
            queue=self._queue,
            sampler=sampler,
            fetcher=self._fetcher,
            start_sample_condition=start_sample_condition,
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
