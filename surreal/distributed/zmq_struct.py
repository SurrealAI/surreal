import zmq
import threading
import surreal.utils as U


def _get_serializer(is_pyobj):
    if is_pyobj:
        return U.serialize
    else:
        return lambda x: x


def _get_deserializer(is_pyobj):
    if is_pyobj:
        return U.deserialize
    else:
        return lambda x: x


class ZmqPushClient(object):
    """
    agent -> replay
    """
    def __init__(self, host, port, is_pyobj=True):
        if host == 'localhost':
            host = '127.0.0.1'
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.set_hwm(42)  # a small magic number to avoid congestion
        address = "tcp://{}:{}".format(host, port)
        print('Pusing to {}'.format(address))
        self.socket.connect(address)
        self._serialize = _get_serializer(is_pyobj)

    def push(self, obj):
        self.socket.send(self._serialize(obj))


class ZmqPullServer(object):
    """
    replay <- agent
    """
    def __init__(self, address, is_pyobj=True):
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.set_hwm(42)  # a small magic number to avoid congestion
        print('Pulling from {}'.format(address))
        self.socket.connect(address)
        self._deserialize = _get_deserializer(is_pyobj)

    def pull(self):
        return self._deserialize(self.socket.recv())


class ZmqServerWorker(threading.Thread):
    """
    replay -> learner, replay handling learner's requests
    """
    def __init__(self, context, handler, is_pyobj):
        threading.Thread.__init__(self)
        self.context = context
        self._handler = handler
        self._serialize = _get_serializer(is_pyobj)
        self._deserialize = _get_deserializer(is_pyobj)
        self.serialize_time = U.TimeRecorder()

    def run(self):
        socket = self.context.socket(zmq.REP)
        socket.connect('inproc://worker')
        while True:
            req = socket.recv()
            res = self.process(req)
            socket.send(res)
        socket.close()

    def process(self, req):
        request = self._deserialize(req)
        response = self._handler(request)
        with self.serialize_time.time():
            res = self._serialize(response)
        return res


class ZmqServer(threading.Thread):
    """
    replay -> learner, manages ZmqServerWorker pool
    Async REQ-REP server
    """
    def __init__(self, port, handler, host='*', is_pyobj=True, num_workers=5, loadbalanced=False):
        """
        Args:
            port:
            handler: takes the request (pyobj) and sends the response
        """
        threading.Thread.__init__(self)
        self.port = port
        self.host = host
        self.handler = handler
        self.is_pyobj = is_pyobj
        self.num_workers = num_workers
        self.serialize_time = U.TimeRecorder()
        self.loadbalanced = loadbalanced

    def run(self):
        context = zmq.Context()
        router = context.socket(zmq.ROUTER)
        address = "tcp://{}:{}".format(self.host, self.port)
        print('Listening on {}'.format(address))
        if self.loadbalanced:
            router.connect(address)
        else:
            router.bind(address)

        dealer = context.socket(zmq.DEALER)
        dealer.bind("inproc://worker")

        workers = []
        for worker_id in range(self.num_workers):
            worker = ZmqServerWorker(context, self.handler, self.is_pyobj)
            worker.start()
            workers.append(worker)

        self.serialize_time = workers[0].serialize_time
        
        # http://zguide.zeromq.org/py:mtserver
        # http://api.zeromq.org/3-2:zmq-proxy
        # **WARNING**: zmq.proxy() must be called AFTER the threads start,
        # otherwise the program hangs!
        # Before calling zmq_proxy() you must set any socket options, and
        # connect or bind both frontend and backend sockets.
        zmq.proxy(router, dealer)
        # Loops
        

        # Never reach
        router.close()
        dealer.close()
        context.term()


class ZmqClientTask(threading.Thread):
    """
    learner <- replay
    Forwards input from 'tasks' to 'out' and hand over response to @handler
    """
    def __init__(self, context, identifier, host, port, handler, is_pyobj):
        threading.Thread.__init__(self)
        self.context = context
        self.id = identifier
        self.address = "tcp://{}:{}".format(host, port)
        print('Requesting to {}'.format(self.address))
        self._handler = handler
        self._serialize = _get_serializer(is_pyobj)
        self._deserialize = _get_deserializer(is_pyobj)

    def run(self):
        out = self.context.socket(zmq.REQ)
        out.connect(self.address)

        tasks = self.context.socket(zmq.REQ)
        tasks.connect("inproc://worker")
        while True:
            tasks.send(b'ready')
            request = tasks.recv()
            
            out.send(request)
            response = out.recv()
            ret = self._deserialize(response)
            self._handler(ret)

        # Never reach
        out.close()
        tasks.close()


class ZmqClientPool(threading.Thread):
    """
    learner <- replay, pull from replay, manages ZmqClientTasks
    """
    def __init__(self, host, port, request, handler, is_pyobj=True, num_workers=5):
        threading.Thread.__init__(self)
        if host == 'localhost':
            host = '127.0.0.1'
        self.host = host
        self.port = port
        self.request = _get_serializer(is_pyobj)(request)
        self.handler = handler
        self.is_pyobj = is_pyobj
        self.num_workers = num_workers

    def run(self):
        context = zmq.Context()
        router = context.socket(zmq.ROUTER)
        router.bind("inproc://worker")

        workers = []
        for worker_id in range(self.num_workers):
            worker = ZmqClientTask(context,
                                    'worker-{}'.format(worker_id),
                                    self.host,
                                    self.port,
                                    self.handler,
                                    self.is_pyobj)
            worker.start()
            workers.append(worker)

        # Distribute all tasks 
        while True:
            address, empty, ready = router.recv_multipart()
            router.send_multipart([address, b'', self.request])

        # Never reach
        router.close()
        context.term()


class ZmqClient(object):
    """
    Just a REQ, connects to ROUTER or REP
    agent <- ps
    """
    def __init__(self, host, port, is_pyobj=True):
        if host == 'localhost':
            host = '127.0.0.1'
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://{}:{}".format(host, port))
        self._serialize = _get_serializer(is_pyobj)
        self._deserialize = _get_deserializer(is_pyobj)

    def request(self, request):
        self.socket.send(self._serialize(request))
        response = self.socket.recv()
        ret = self._deserialize(response)
        return ret


class ZmqPublishServer(object):
    """
    PUB-SUB pattern: PUB server
    learner -> ps
    """
    def __init__(self, port, is_pyobj=True):
        """
        Args:
            port:
        """
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.set_hwm(1)  # aggressively drop late messages
        self.socket.bind("tcp://*:{}".format(port))
        self._thread = None
        self._serialize = _get_serializer(is_pyobj)

    def publish(self, data, topic):
        topic = U.str2bytes(topic)
        data = self._serialize(data)
        self.socket.send_multipart([topic, data])


class ZmqSubscribeClient(object):
    """
    Simple REQ-REP server
    ps <- learner
    """
    def __init__(self, host, port, handler, topic, is_pyobj=True):
        """
        Args:
            port:
            handler: processes the data received from SUB
        """
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        if host == 'localhost':
            host = '127.0.0.1'
        self.socket.set_hwm(1)  # aggressively drop late messages
        self.socket.connect("tcp://{}:{}".format(host, port))
        topic = U.str2bytes(topic)
        self.socket.setsockopt(zmq.SUBSCRIBE, topic)
        self._handler = handler
        self._thread = None
        self._deserialize = _get_deserializer(is_pyobj)

    def _run_loop(self):
        while True:
            _, data = self.socket.recv_multipart()
            self._handler(self._deserialize(data))

    def run_loop(self, block):
        if block:
            self._run_loop()
        else:
            if self._thread:
                raise RuntimeError('loop already running')
            self._thread = U.start_thread(self._run_loop)
            return self._thread


class ZmqQueue(object):
    """
    Replay side
    """
    def __init__(self,
                 address,
                 max_size,
                 is_pyobj=True,
                 start_thread=True):
        """

        Args:
            port:
            max_size:
            start_thread:
            is_pyobj: pull and convert to python object
        """
        self._puller = ZmqPullServer(address=address, is_pyobj=is_pyobj)
        self._queue = U.FlushQueue(max_size=max_size)
        # start
        self._enqueue_thread = None
        if start_thread:
            self.start_enqueue_thread()

    def start_enqueue_thread(self):
        if self._enqueue_thread is not None:
            raise RuntimeError('enqueue_thread already started')
        self._enqueue_thread = U.start_thread(self._run_enqueue)
        return self._enqueue_thread

    def _run_enqueue(self):
        while True:
            self._queue.put(self._puller.pull())
            # del obj  # clear the last obj's ref count

    def get(self):
        return self._queue.get(block=True, timeout=None)

    def size(self):
        return len(self._queue)

    __len__ = size

