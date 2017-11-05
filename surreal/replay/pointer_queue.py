import queue
from surreal.distributed import ExpPack, RedisClient
from surreal.utils.common import StoppableThread


class _PointerDequeueThread(StoppableThread):
    def __init__(self, queue, handler):
        self.queue = queue
        self.handler = handler
        super().__init__()

    def run(self):
        while True:
            if self.is_stopped():
                break
            exp = self.queue.get(block=True, timeout=None)
            self.handler(exp.data)
            self.queue.task_done()


class PointerQueue:
    def __init__(self, redis_client, queue_name, maxsize=0):
        self.queue = queue.Queue(maxsize=maxsize)
        assert isinstance(redis_client, RedisClient)
        self.queue_name = queue_name
        self.client = redis_client
        self._dequeue_thread = None

    def _enqueue_producer(self, binary, i):
        return self.queue.put(ExpPack.deserialize(binary))

    def start_enqueue_thread(self):
        """
        Producer thread, get pointerpacks queued up in list from Redis
        """
        return self.client.pull_from_queue_thread(
            queue_name=self.queue_name,
            handler=self._enqueue_producer
        )

    def stop_enqueue_thread(self):
        self.client.stop_queue_thread(self.queue_name)

    def start_dequeue_thread(self, handler):
        """
        handler function takes an experience dict (ExpPack.data) and
        inserts it into a priority replay data structure.
        """
        if self._dequeue_thread is not None:
            raise ValueError('Dequeue thread is already running')
        self._dequeue_thread = _PointerDequeueThread(self.queue, handler)
        self._dequeue_thread.start()
        return self._dequeue_thread

    def stop_dequeue_thread(self):
        self._dequeue_thread.stop()
        self._dequeue_thread = None


