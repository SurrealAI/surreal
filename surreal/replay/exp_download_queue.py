import queue
import itertools
from time import sleep
from surreal.comm import RedisClient, ExpPack
from surreal.utils.common import StoppableThread
from .obs_downloader import ObsDownloader


class _EnqueueThread(StoppableThread):
    def __init__(self, queue, sampler, downloader, start_sample_condition):
        """
        start_sample_condition(): begins sampling only when this returns True.
            Example: when the replay memory exceeds a threshold size
        """
        self.queue = queue
        self.sampler = sampler
        self.downloader = downloader
        self.start_sample_condition = start_sample_condition
        super().__init__()

    def run(self):
        while not self.start_sample_condition():
            if self.is_stopped():
                break
            sleep(0.5)
        for i in itertools.count():
            if self.is_stopped():
                break
            exp_dicts = self.sampler(i)
            exps = self.downloader.download(exp_dicts)
            # block if the queue is full
            self.queue.put(exps, block=True, timeout=None)


class ExpDownloadQueue:
    def __init__(self, redis_client, maxsize):
        self.queue = queue.Queue(maxsize=maxsize)
        assert isinstance(redis_client, RedisClient)
        self.downloader = ObsDownloader(redis_client)
        self._enqueue_thread = None

    def start_enqueue_thread(self, sampler, start_sample_condition):
        """
        Producer thread, runs sampler function on a priority replay structure
        Args:
            sampler: batch_i -> list of exp_dicts with 'obs_pointers' field
        """
        if self._enqueue_thread is not None:
            raise ValueError('Enqueue thread is already running')
        self._enqueue_thread = _EnqueueThread(
            queue=self.queue,
            sampler=sampler,
            downloader=self.downloader,
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
        return self.queue.get(block=True, timeout=None)
