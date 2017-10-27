import time
import threading
from surreal.comm import RedisClient
from .pointer_queue import PointerQueue
from .exp_download_queue import ExpDownloadQueue


class UniformReplay(object):
    def __init__(self, redis_client, batch_size):
        assert isinstance(redis_client, RedisClient)
        self.pointer_queue = PointerQueue(
            redis_client=redis_client,
            queue_name='replay',
        )
        self.exp_download_queue = ExpDownloadQueue(
            redis_client=redis_client,
            maxsize=5
        )
        self.memory = {}
        self.batch_size = batch_size
        self._lock = threading.Lock()

    def insert(self, exp_dict):
        print('INSERT', exp_dict)
        time.sleep(0.5)
        self.memory[len(self.memory)] = exp_dict

    def sample(self, batch_i):
        # TODO: self.batch_size
        samps = []
        print('SAMPLE total size', len(self.memory), 'batch_i', batch_i)
        for i in self.memory:
            time.sleep(.3)
            if i % 3 == 0:
                samps.append(self.memory[i])
        return samps

    def start_sample_condition(self):
        return len(self.memory) > 10

    def _locked_insert(self, *args, **kwargs):
        """
        Must not sample and insert at the same time
        """
        with self._lock:
            return self.insert(*args, **kwargs)

    def _locked_sample(self, *args, **kwargs):
        with self._lock:
            return self.sample(*args, **kwargs)

    def start_threads(self):
        self.pointer_queue.run_enqueue_thread()
        self.pointer_queue.run_dequeue_thread(self._locked_insert)
        self.exp_download_queue.run_enqueue_thread(
            self._locked_sample,
            self.start_sample_condition,
        )

    def next_batch(self):
        # TODO method to aggregate exp_dicts into batched tensors
        return self.exp_download_queue.dequeue()

    def batch_iter(self):
        while True:
            yield self.next_batch()
