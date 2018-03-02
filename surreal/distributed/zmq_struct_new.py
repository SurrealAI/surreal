import queue
import zmq
from threading import Thread
from multiprocessing import Process
import surreal.utils as U
from tensorplex import Logger
from surreal.distributed.inmemory import inmem_serialize, inmem_deserialize, inmem_dump
from surreal.distributed.zmq_struct import zmq_logger
# zmq_logger = Logger.get_logger(
#     'zmq',
#     stream='stdout',
#     time_format='hms',
#     show_level=True,
# )

class ZmqSocketWrapper(object):
    """
        Wrapper around zmq socket, manages resources automatically
    """
    def __init__(self, mode, bind, address=None, host=None, port=None, context=None):
        """
        Args:
            @host: specifies address, localhost is translated to 127.0.0.1
            @address
            @port: specifies address
            @mode: zmq.PUSH, zmq.PULL, etc.
            @bind: True -> bind to address, False -> connect to address (see zmq)
            @context: Zmq.Context object, if None, client creates its own context
            @name: Purely for logging purposes
            @hwm
        """
        if address is not None:
            self.address = address
        else:
            # https://stackoverflow.com/questions/6024003/why-doesnt-zeromq-work-on-localhost
            assert host is not None and port is not None
            # Jim's note, ZMQ does not like localhost
            if host == 'localhost':
                host = '127.0.0.1'
            self.address = "tcp://{}:{}".format(host, port)

        if context is None:
            self.context = zmq.Context()
            self.owns_context = True
        else:
            self.context = context
            self.owns_context = False

        self.mode = mode
        self.bind = bind
        self.socket = self.context.socket(self.mode)
        self.established = False

    def establish(self):
        """
            We want to allow subclasses to configure the socket before connecting
        """
        if self.established:
            raise RuntimeError('Trying to establish a socket twice')
        self.established = True
        if self.bind:
            zmq_logger.infofmt('[{}] binding to {}', self.socket_type, self.address)
            self.socket.bind(self.address)
        else:
            zmq_logger.infofmt('[{}] connecting to {}', self.socket_type, self.address)
            self.socket.connect(self.address)
        return self.socket

    def __del__(self):
        if self.established:
            self.socket.close()
        if self.owns_context: # only terminate context when we created it
            self.context.term()

    @property
    def socket_type(self):
        if self.mode == zmq.PULL:
            return 'PULL'
        elif self.mode == zmq.PUSH:
            return 'PUSH'
        elif self.mode == zmq.PUB:
            return 'PUB'
        elif self.mode == zmq.SUB:
            return 'SUB'
        elif self.mode == zmq.PAIR:
            return 'PAIR'
        elif self.mode == zmq.REQ:
            return 'REQ'
        elif self.mode == zmq.REP:
            return 'REP'
        elif self.mode == zmq.ROUTER:
            return 'ROUTER'
        elif self.mode == zmq.DEALER:
            return 'DEALER'


class ZmqPusher(ZmqSocketWrapper):
    def __init__(self, host, port, preprocess=None, hwm=42):
        super().__init__(host=host, port=port, mode=zmq.PUSH, bind=False)
        self.socket.set_hwm(hwm)
        self.preprocess = preprocess
        self.establish()

    def push(self, data):
        if self.preprocess:
            data = self.preprocess(data)
        self.socket.send(data)


class ZmqPuller(ZmqSocketWrapper):
    def __init__(self, host, port, bind, preprocess=None):
        super().__init__(host=host, port=port, mode=zmq.PULL, bind=bind)
        self.preprocess = preprocess
        self.establish()

    def pull(self):
        data = self.socket.recv()
        if self.preprocess:
            data = self.preprocess(data)
        return data


class ZmqPullerThread(Thread):
    def __init__(self, host, port, bind, handler):
        Thread.__init__(self)
        self.socket = ZmqSocketWrapper(host=host, port=port, mode=zmq.PULL, bind=bind)

    def run(self):
        self.socket.establish()
        while True:
            data = self.socket.recv()
            self.handler(data)


class OutsourceTask(Process):
    def __init__(self):
        """
            Override this to setup states in the main process
        """
        Process.__init__(self)
        pass

    def run(self):
        # You must initilize the transmission channel AFTER you fork off
        self.pusher = ZmqPusher(self.host, self.port, preprocess=inmem_serialize, hwm=5)

    def setup_comm(self, host, port):
        """
            Setup the communication, use self.pusher.push() to send information back to manager
        """
        self.host = host
        self.port = port


class OutsourceManager(Thread):
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
        self.puller = ZmqPuller(host=self.host, 
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
            self.handler(self.puller.pull())

        # Will never reach here
        for worker in self.workers:
            worker.join()


class OutsourceManagerQueue(OutsourceManager):
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


class DataBatchFetchingTask(OutsourceTask):
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

    def run(self):
        self.pusher = ZmqPusher(self.host, self.port, preprocess=inmem_serialize, hwm=5)
        self.pool.start()
        self.pool.join()

    def handler(self, data):
        data = U.deserialize(data)
        self.pusher.push(data)


class LearnerDataPrefetcher(OutsourceManagerQueue):
    """
        Convenience class that initializes everything from session config
        + batch_size
    """
    def __init__(self, session_config, batch_size):
        self.sampler_host = session_config.replay.sampler_frontend_host
        self.sampler_port = session_config.replay.sampler_frontend_port
        self.batch_size = batch_size
        self.max_queue_size = session_config.learner.max_prefetch_batch_queue
        self.prefetch_host = session_config.learner.prefetch_host
        self.prefetch_port = session_config.learner.prefetch_port
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

        
class ZmqReqWorker(Thread):
    """
        Requests to 'inproc://worker' to get request data
        Sends requests to 'tcp://@host:@port'
        Gives response to @handler
    """
    def __init__(self, context, host, port, handler):
        Thread.__init__(self)
        self.context = context
        
        self.sw_out = ZmqSocketWrapper(host=host, port=port, 
            mode=zmq.REQ, bind=False, context=context)

        self.sw_inproc = ZmqSocketWrapper(address='inproc://worker', mode=zmq.REQ,
                                        bind=False, context=context)
        self.handler = handler

    def run(self):
        self.out_socket = self.sw_out.establish()
        self.task_socket = self.sw_inproc.establish()
        while True:
            self.task_socket.send(b'ready')
            request = self.task_socket.recv()
            
            self.out_socket.send(request)
            response = self.out_socket.recv()
            self.handler(response)

        # Never reaches here
        self.out_socket.close()
        self.task_socket.close()

class ZmqReqClientPool(Thread):
    """
        Spawns num_workers threads and send requests to the provided endpoint
        Responses are given to @handler
    """
    def __init__(self, host, port, handler, num_workers=5):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.handler = handler
        self.num_workers = num_workers

    def get_request(self):
        raise NotImplementedError

    def run(self):
        context = zmq.Context()
        router = context.socket(zmq.ROUTER)
        router.bind("inproc://worker")

        workers = []
        for worker_id in range(self.num_workers):
            worker = ZmqReqWorker(context,
                                    self.host,
                                    self.port,
                                    self.handler)
            worker.start()
            workers.append(worker)

        # Distribute all tasks 
        while True:
            request = self.get_request()
            address, empty, ready = router.recv_multipart()
            router.send_multipart([address, b'', request])

        # Never reach
        router.close()
        context.term()

class ZmqReqClientPoolFixedRequest(ZmqReqClientPool):
    """
        Always blasts the same request
    """
    def __init__(self, host, port, handler, request, num_workers=5):
        super().__init__(host, port, handler, num_workers)
        self.request = request

    def get_request(self):
        return self.request



# class ZmqReq(ZmqSocketWrapper):
#     def __init__(self, host, port, serializer=None):
#         super().__init__(host=host, port=port, mode=zmq.REQ, bind=False)
#         self.serializer = serializer

#     def req(self, data):
#         if self.serializer:
#             data = self.serializer(data)
#         self.socket.send(data)


# class ZmqRep(ZmqSocketWrapper):
#     def __init__(self, host, port, bind, serializer=None):
#         super().__init__(host=host, port=port, mode=zmq.REP, bind=bind)
#         self.serializer = serializer

#     def recv(self):
#         data = self.socket.recv()
#         if self.serializer:
#             data = self.serializer(data)
#         return data


# class ZmqPub(ZmqSocketWrapper):
#     def __init__(self, host, port, hwm=1, serializer=None):
#         super().__init__(host=host, port=port, mode=zmq.PUB, bind=True)
#         self.socket.set_hwm(hwm)
#         self.serializer = serializer


#     def pub(self, topic, data):
#         topic = U.str2bytes(topic)
#         if self.serializer:
#             data = self.serializer(data)
#         self.socket.send_multipart([topic, data])


class ZmqSub(ZmqSocketWrapper):
    def __init__(self, host, port, topic, hwm=1, deserializer=None):
        super().__init__(host=host, port=port, mode=zmq.SUB, bind=False)
        self.socket.set_hwm(hwm)
        self.socket.setsockopt(zmq.SUBSCRIBE, topic)
        self.establish()

    def recv(self):
        _, data = self.socket.recv_multipart()
        return data


class SubClient(Thread):
    def __init__(self, host, port, topic, hanlder, deserializer=None, hwm=1):
        Thread.__init__(self)
        if deserializer is None:
            deserializer = U.default_deserializer
        self.socket_wrapper = ZmqSub(host, port, topic, hwm)
        self.host = host
        self.port = port
        self.topic = topic
        self.deserializer = deserializer
        self.handler = handler

    def run(self):
        zmq_logger.infofmt('SubClient listening for topic {} on {}:{}', 
                            self.host, self.port, self.topic)
        while True:
            data = self.sub.recv()
            self.handler(data)


class FlushPullQueue(Thread):
    """
    Replay side
    """
    def __init__(self,
                 host,
                 port,
                 max_size,
                 deserializer
                ):
        """

        Args:
            port:
            max_size:
            start_thread:
            is_pyobj: pull and convert to python object
        """
        self.puller = ZmqPull(host=host, port=port, deserializer=deserializer)
        self._queue = U.FlushQueue(max_size=max_size)

    def start_enqueue_thread(self):
        if self._enqueue_thread is not None:
            raise RuntimeError('enqueue_thread already started')
        self._enqueue_thread = U.start_thread(self._run_enqueue)
        return self._enqueue_thread

    def run(self):
        while True:
            self._queue.put(self.puller.pull())

    def get(self):
        return self._queue.get(block=True, timeout=None)

    def size(self):
        return len(self._queue)

    __len__ = size

