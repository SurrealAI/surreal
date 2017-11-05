import itertools
import threading

import surreal.utils as U
from surreal.distributed import RedisClient
from surreal.distributed.exp_fetcher_queue import ExpFetcherQueue
from surreal.distributed.pointer_queue import PointerQueue


class Replay(object):
    def __init__(self, *,
                 redis_client,
                 batch_size,
                 name='replay',
                 download_queue_size=5):
        U.assert_type(redis_client, RedisClient)
        self.batch_size = batch_size
        self._pointer_queue = PointerQueue(
            redis_client=redis_client,
            queue_name=name,
        )
        self._exp_fetcher_queue = ExpFetcherQueue(
            redis_client=redis_client,
            maxsize=download_queue_size,
        )
        self._lock = threading.Lock()

    def insert(self, exp_dict):
        """
        Add a new experience to the replay.
        Args:
            exp_dict: experience dictionary with
                {"obs_pointers", "reward", "action", "info"} keys
        """
        raise NotImplementedError

    def sample(self, batch_size, batch_i):
        """
        This function is called in the `exp_download_queue` thread, its operation
        is async, i.e. overlaps with the insertion operations.

        Args:
            batch_size: passed from self.batch_size, defined in the
                constructor upfront.
            batch_i: the i-th batch it is sampling.
            Note that `batch_size` is

        Returns:
            a list of exp_dicts
        """
        raise NotImplementedError

    def start_sample_condition(self):
        """
        Tells the thread to start sampling only when this condition is met.
        For example, only when the replay memory has > 10K experiences.

        Returns:
            bool: whether to start sampling or not
        """
        raise NotImplementedError

    def aggregate_batch(self, exp_list):
        """
        Will be called in `next_batch()` method to produce the actual inputs
        to the neural network training loop.

        Args:
            exp_list: list of experience dictionaries with actual observations
                {"obses", "reward", "action", "info"} keys

        Returns:
            batched Tensors, batched action/reward vectors, etc.
        """
        raise NotImplementedError

    def _locked_insert(self, exp_dict):
        """
        Must not sample and insert at the same time
        """
        with self._lock:
            return self.insert(exp_dict)

    def _locked_sample(self, batch_i):
        with self._lock:
            return self.sample(self.batch_size, batch_i)

    def start_threads(self):
        """
        Call this method to launch all background threads that talk to Redis.
        """
        self._pointer_queue.start_enqueue_thread()
        self._pointer_queue.start_dequeue_thread(self._locked_insert)
        self._exp_fetcher_queue.start_enqueue_thread(
            self._locked_sample,
            self.start_sample_condition,
        )

    def stop_threads(self):
        self._pointer_queue.stop_enqueue_thread()
        self._pointer_queue.stop_dequeue_thread()
        self._exp_fetcher_queue.stop_enqueue_thread()

    def next_batch(self):
        exp_list = self._exp_fetcher_queue.dequeue()
        return self.aggregate_batch(exp_list)

    def batch_iterator(self):
        """
        Yields:
            batch_i, (batched inputs to neural network)
        """
        for batch_i in itertools.count():
            yield batch_i, self.next_batch()
