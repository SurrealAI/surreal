import queue
from multiprocessing import Process
import surreal.utils as U
from surreal.distributed.zmq_struct import ZmqSender, ZmqReceiver, ZmqReqClientPoolFixedRequest
from threading import Thread, Lock
from surreal.distributed.inmemory import inmem_serialize, inmem_deserialize, inmem_dump
import os


class MultiprocessTask(Process):
    def __init__(self):
        """
            Override this to setup states in the main process
        """
        Process.__init__(self)
        pass

    def run(self):
        # You must initilize the transmission channel AFTER you fork off
        self.sender = ZmqSender(self.host, self.port, preprocess=inmem_serialize)
        # self.pusher = ZmqPusher(self.host, self.port, preprocess=inmem_serialize, hwm=5)

    def setup_comm(self, host, port):
        """
            Setup the communication, use self.pusher.push() to send information back to manager
        """
        self.host = host
        self.port = port


class MultiprocessManager(Thread):
    def __init__(self, host, port, task, n_workers, handler, args=None, kwargs=None):
        Thread.__init__(self)
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        self.n_workers = n_workers
        self.host = host
        self.port = port
        self.handler = handler
        self.task = task
        self.args = args
        self.kwargs = kwargs
        

    def run(self):
        # Use receiver here to rate-limit the workers, using pull-push involves a large cache
        # https://stackoverflow.com/questions/22613737/how-could-i-set-hwm-in-the-push-pull-pattern-of-zmq
        self.receiver = ZmqReceiver(host=self.host, 
                                port=self.port, 
                                bind=True, 
                                preprocess=inmem_deserialize)

        self.workers = []
        for i in range(self.n_workers):
            worker = self.task(*self.args, **self.kwargs)
            worker.setup_comm(self.host, self.port)
            worker.start()
            self.workers.append(worker)

        while True:
            self.handler(self.receiver.recv())

        # Will never reach here
        for worker in self.workers:
            worker.join()


class MultiprocessManagerQueue(MultiprocessManager):
    """
        Uses a queue to store response data instead of calling a handler
    """
    def __init__(self, host, port, task, n_workers, max_queue_size=10, args=None, kwargs=None):
        super().__init__(host=host, 
                         port=port,
                         task=task, 
                         n_workers=n_workers, 
                         handler=self.put, 
                         args=args, 
                         kwargs=kwargs)
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.timer = U.TimeRecorder()

    def __len__(self):
        return len(self.queue)

    def get(self):
        """
            Returns a SharedMemoryObject
            whose .data attribute contains data
        """
        with self.timer.time():
            return self.queue.get(block=True)

    def put(self, data):
        self.queue.put(data, block=True)


class DataBatchFetchingTask(MultiprocessTask):
    def __init__(self, host, port, request, num_workers):
        """
            Override this to setup states in the main process
        """
        super().__init__()
        self.pool = ZmqReqClientPoolFixedRequest(host=host, 
                                                port=port, 
                                                request=request,
                                                handler=self.handler,
                                                num_workers=num_workers)
        # Client pool can get multiple responses every time, 
        # zmq sockets are not threadsafe
        self.lock = Lock()

    def run(self):
        self.sender = ZmqSender(self.host, self.port, preprocess=inmem_serialize)
        self.pool.start()
        self.pool.join()

    def handler(self, data):
        data = U.deserialize(data)
        with self.lock:
            self.sender.send(data)


class LearnerDataPrefetcher(MultiprocessManagerQueue):
    """
        Convenience class that initializes everything from session config
        + batch_size
    """
    def __init__(self, session_config, batch_size):
        self.sampler_host = os.environ['SYMPH_SAMPLER_FRONTEND_HOST']
        self.sampler_port = os.environ['SYMPH_SAMPLER_FRONTEND_PORT']
        self.batch_size = batch_size
        self.max_queue_size = session_config.learner.max_prefetch_batch_queue
        self.prefetch_host = 'localhost'
        self.prefetch_port = os.environ['SYMPH_PREFETCH_QUEUE_PORT']
        self.prefetch_processes = session_config.learner.prefetch_processes
        self.prefetch_threads_per_process = session_config.learner.prefetch_threads_per_process

        task = DataBatchFetchingTask
        kwargs = {  'host': self.sampler_host,
                    'port': self.sampler_port,
                    'request': U.serialize(batch_size),
                    'num_workers': self.prefetch_threads_per_process
                 }
        super().__init__(host=self.prefetch_host, 
                        port=self.prefetch_port, 
                        task=task, 
                        n_workers=self.prefetch_processes,
                        kwargs=kwargs,
                        max_queue_size=self.max_queue_size)

        