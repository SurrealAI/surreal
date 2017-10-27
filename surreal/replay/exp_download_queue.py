import queue
import itertools
from surreal.comm import PointerPack, RedisClient
from surreal.utils.common import StoppableThread
from .exp_downloader import ExpDownloader


class _ExpEnqueueThread(StoppableThread):
    def __init__(self, queue, sampler):
        self.queue = queue
        self.sampler = sampler
        super().__init__()

    def run(self):
        for i in itertools.count():
            if self.is_stopped():
                break
            # block if the queue is full
            self.queue.put(self.sampler(i), block=True, timeout=None)


class _ExpDequeueThread(StoppableThread):
    def __init__(self, queue, handler):
        self.queue = queue
        self.handler = handler
        super().__init__()

    def run(self):
        while True:
            if self.is_stopped():
                break
            pointerpack = self.queue.get(block=True, timeout=None)
            self.handler(pointerpack)
            self.queue.task_done()


class ExpDownloadQueue:
    def __init__(self, redis_client, maxsize):
        self.queue = queue.Queue(maxsize=maxsize)
        assert isinstance(redis_client, RedisClient)
        self.downloader = ExpDownloader(redis_client)
        self._dequeue_thread = None

    def _enqueue_producer(self, binary, i):
        return self.queue.put(PointerPack.deserialize(binary))

    def enqueue_thread(self, sampler):
        """
        Producer thread, runs sampler function on a priority replay structure
        Args:
            sampler: batch_i -> list of
                       {'exp_pointer': "hashkey", 'obs_pointers': ["hashkey"s]}
        """
        return self.client.pull_from_queue_thread(
            queue_name=self.queue_name,
            handler=self._enqueue_producer
        )

    def stop_enqueue_thread(self):
        self.client.stop_queue_thread(self.queue_name)

    def dequeue_thread(self, handler):
        """
        handler function takes a PointerPack and processes it
        """
        if self._dequeue_thread is not None:
            raise ValueError('Dequeue thread is already running')
        self._dequeue_thread = _DownloadDequeueThread(self.queue, handler)
        self._dequeue_thread.start()
        return self._dequeue_thread

    def stop_dequeue_thread(self):
        self._dequeue_thread.stop()
        self._dequeue_thread = None


