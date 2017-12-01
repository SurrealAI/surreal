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
    def __init__(self, host, port, is_pyobj=True):
        if host == 'localhost':
            host = '127.0.0.1'
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.set_hwm(42)  # a small magic number to avoid congestion
        self.socket.connect("tcp://{}:{}".format(host, port))
        self._serialize = _get_serializer(is_pyobj)

    def push(self, obj):
        self.socket.send(self._serialize(obj))


class ZmqPullServer(object):
    def __init__(self, port, is_pyobj=True):
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.set_hwm(42)  # a small magic number to avoid congestion
        self.socket.bind("tcp://127.0.0.1:{}".format(port))
        self._deserialize = _get_deserializer(is_pyobj)

    def pull(self):
        return self._deserialize(self.socket.recv())


class ZmqServer(object):
    """
    Simple REQ-REP server
    """
    def __init__(self, port, handler, is_pyobj=True):
        """
        Args:
            port:
            handler: takes the request (pyobj) and sends the response
        """
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://127.0.0.1:{}".format(port))
        self._handler = handler
        self._thread = None
        self._serialize = _get_serializer(is_pyobj)
        self._deserialize = _get_deserializer(is_pyobj)

    def _run_loop(self):
        while True:
            request = self._deserialize(self.socket.recv())
            response = self._handler(request)
            self.socket.send(self._serialize(response))

    def run_loop(self, block):
        if block:
            self._run_loop()
        else:
            if self._thread:
                raise RuntimeError('loop already running')
            self._thread = U.start_thread(self._run_loop)
            return self._thread


class ZmqClient(object):
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
        return self._deserialize(self.socket.recv())


class ZmqPublishServer(object):
    """
    PUB-SUB pattern: PUB server
    """
    def __init__(self, port, is_pyobj=True):
        """
        Args:
            port:
        """
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.set_hwm(1)  # aggressively drop late messages
        self.socket.bind("tcp://127.0.0.1:{}".format(port))
        self._thread = None
        self._serialize = _get_serializer(is_pyobj)

    def publish(self, data, topic):
        topic = U.str2bytes(topic)
        data = self._serialize(data)
        self.socket.send_multipart([topic, data])


class ZmqSubscribeClient(object):
    """
    Simple REQ-REP server
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
                 port,
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
        self._puller = ZmqPullServer(port=port, is_pyobj=is_pyobj)
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

