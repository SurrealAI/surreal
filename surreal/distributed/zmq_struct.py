import zmq
import threading
import surreal.utils as U


class ZmqPusherClient(object):
    def __init__(self, host, port):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.set_hwm(42)  # a small magic number to avoid congestion
        self.socket.connect("tcp://{}:{}".format(host, port))

    def push(self, binary):
        self.socket.send(binary)

    def push_pyobj(self, obj):
        self.socket.send(U.serialize(obj))


class DummyPusherClient(object):  # debugging only
    def __init__(self, host, port):
        pass

    def push(self, binary):
        print('PUSHED')

    def push_pyobj(self, obj):
        print('PUSHED PYOBJ')

# ZmqPusherClient = DummyPusherClient


class ZmqPullerServer(object):
    def __init__(self, port):
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.set_hwm(42)  # a small magic number to avoid congestion
        self.socket.bind("tcp://127.0.0.1:{}".format(port))

    def pull(self):
        return self.socket.recv()

    def pull_pyobj(self):
        return U.deserialize(self.socket.recv())


class ZmqQueue(object):
    """
    Replay side
    """
    def __init__(self,
                 port,
                 max_size,
                 start_thread=True,
                 is_pyobj=True):
        """

        Args:
            port:
            max_size:
            start_thread:
            is_pyobj: pull and convert to python object
        """
        self._puller = ZmqPullerServer(port=port)
        self._queue = U.FlushQueue(max_size=max_size)
        # start
        self._to_pyobj = is_pyobj
        self._enqueue_thread = None
        if start_thread:
            self.start_enqueue_thread()

    def start_enqueue_thread(self):
        if self._enqueue_thread is not None:
            raise RuntimeError('enqueue_thread already started')
        self._enqueue_thread = threading.Thread(target=self._run_enqueue)
        self._enqueue_thread.daemon = True
        self._enqueue_thread.start()
        return self._enqueue_thread

    def _run_enqueue(self):
        while True:
            if self._to_pyobj:
                obj = self._puller.pull_pyobj()
            else:
                obj = self._puller.pull()
            self._queue.put(obj)
            del obj  # clear the last obj's ref count

    def get(self):
        return self._queue.get(block=True, timeout=None)

    def size(self):
        return len(self._queue)

    __len__ = size


"""
class ZmqQueueClient(object):
    def __init__(self,
                 host,
                 port,
                 batch_interval,
                 use_pickle=True,
                 start_thread=True):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.set_hwm(100)
        self.socket.connect("tcp://{}:{}".format(host, port))
        self._use_pickle = use_pickle
        self._batch_interval = batch_interval
        if self._use_pickle:
            self._send = self.socket.send_pyobj
        else:
            self._send = self.socket.send
        self._batch_buffer = []
        self._batch_lock = threading.Lock()

        self.batch_thread = None
        if self._batch_interval > 0 and start_thread:
            self.start_batch_thread()

    def start_batch_thread(self):
        if self.batch_thread is not None:
            raise ValueError('batch_thread already running')
        self.batch_thread = threading.Thread(target=self._run_batch)
        self.batch_thread.daemon = True
        self.batch_thread.start()
        return self.batch_thread

    def _run_batch(self):
        while True:
            if self._batch_buffer:
                with self._batch_lock:
                    self._send(self._batch_buffer)
                    self._batch_buffer.clear()
            time.sleep(self._batch_interval)

    def enqueue(self, obj):
        if self._batch_interval == 0:  # no batching
            self._send(obj)
        else:
            with self._batch_lock:
                self._batch_buffer.append(obj)
"""
