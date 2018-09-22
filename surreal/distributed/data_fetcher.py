import os
import queue
from caraml.zmq import DataFetcher
from benedict import BeneDict
import surreal.utils as U
from threading import Thread


class LearnerDataPrefetcher(DataFetcher):
    """
        Convenience class that initializes everything from session config
        + batch_size

        Fetches data from replay in multiple processes and put them into
        a queue
    """
    def __init__(self,
                 session_config,
                 batch_size,
                 worker_preprocess=None,
                 main_preprocess=None):
        self.max_fetch_queue = session_config.learner.max_prefetch_queue
        self.max_preprocess_queue = session_config.learner.max_preprocess_queue
        self.fetch_queue = queue.Queue(maxsize=self.max_fetch_queue)
        self.preprocess_queue = queue.Queue(maxsize=self.max_preprocess_queue)
        self.timer = U.TimeRecorder()

        self.sampler_host = os.environ['SYMPH_SAMPLER_FRONTEND_HOST']
        self.sampler_port = os.environ['SYMPH_SAMPLER_FRONTEND_PORT']
        self.batch_size = batch_size
        self.prefetch_processes = session_config.learner.prefetch_processes
        self.prefetch_host = '127.0.0.1'
        self.worker_comm_port = os.environ['SYMPH_PREFETCH_QUEUE_PORT']
        self.worker_preprocess = worker_preprocess
        self.main_preprocess = main_preprocess
        super().__init__(
            handler=self._put,
            remote_host=self.sampler_host,
            remote_port=self.sampler_port,
            requests=self.request_generator(),
            worker_comm_port=self.worker_comm_port,
            remote_serializer=U.serialize,
            remote_deserialzer=U.deserialize,
            n_workers=self.prefetch_processes,
            worker_handler=self.worker_preprocess)

    def run(self):
        self._preprocess_thread = Thread(target=self._preprocess_loop,
                                         daemon=True)
        self._preprocess_thread.start()
        super().run()

    def _preprocess_loop(self):
        while True:
            sharedmem_obj = self.fetch_queue.get(block=True)
            batch = BeneDict(sharedmem_obj.data)
            batch = self.main_preprocess(batch)
            self.preprocess_queue.put(batch)

    def _put(self, _, data):
        self.fetch_queue.put(data, block=True)

    def get(self):
        """
            Returns a SharedMemoryObject
            whose .data attribute contains data
        """
        with self.timer.time():
            return self.preprocess_queue.get(block=True)

    def request_generator(self):
        while True:
            yield self.batch_size
