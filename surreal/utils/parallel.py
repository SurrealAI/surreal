import queue
import collections
import threading
import time


def start_thread(func, daemon=True, args=None, kwargs=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    t = threading.Thread(
        target=func,
        args=args,
        kwargs=kwargs,
        daemon=daemon,
    )
    t.start()
    return t


class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        self._stop_event = threading.Event() # stoppable thread
        super().__init__(*args, **kwargs)

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()


class _JobThread(StoppableThread):
    def __init__(self, queue):
        self._queue = queue
        super().__init__()

    def run(self):
        while True:
            if self.is_stopped():
                break
            func, return_placeholder, event = self._queue.get(block=True)
            if func is None:  # sentinel stop
                break
            ret = func()
            return_placeholder[0] = ret
            event.set()


class JobQueue(object):
    def __init__(self, wait_for_result=True):
        """
        Args:
            wait_for_result: if True, self.process(...) will wait and return
                the result, otherwise returns immediately and result will be
                discarded.

        Notes:
            Use a PriorityQueue to jump the queue with more important jobs.
            Use time.time() as a second priority to fall back to FIFO when the
            job priority is the same.
            https://stackoverflow.com/questions/9289614/how-to-put-items-into-priority-queues
        """
        self._queue = queue.Queue()
        self._thread = None
        self._wait_for_result = wait_for_result

    def process(self, func, *args, **kwargs):
        event = threading.Event()
        return_placeholder = [None]
        self._queue.put((lambda: func(*args, **kwargs),
                         return_placeholder,
                         event))
        if self._wait_for_result:
            event.wait()
            return return_placeholder[0]
        else:
            return None

    def start_thread(self):
        """
        Continuously execute jobs in FIFO order
        """
        if self._thread is not None:
            raise RuntimeError('JobQueue thread already running')
        self._thread = _JobThread(self._queue)
        self._thread.start()
        return self._thread

    def stop_thread(self):
        self._queue.put((None, None, None))  # sentinel
        t = self._thread
        t.stop()
        self._thread = None
        return t


class FlushQueue(object):
    """
    Handles a continuous stream of incoming data. When max capacity is reached,
    flush out the oldest data that didn't have time to be processed.
    Adapted from python's queue.Queue source code.
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = collections.deque(maxlen=max_size)

        # mutex must be held whenever the queue is mutating.  All methods
        # that acquire mutex must release it before returning.  mutex
        # is shared between the three conditions, so acquiring and
        # releasing the conditions also acquires and releases mutex.
        self._mutex = threading.Lock()

        # Notify not_empty whenever an item is added to the queue; a
        # thread waiting to get is notified then.
        self._not_empty = threading.Condition(self._mutex)

    def put(self, item):
        '''
        Put an item into the queue.

        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until a free slot is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot
        is immediately available, else raise the Full exception ('timeout'
        is ignored in that case).
        '''
        with self._not_empty:
            self.queue.append(item)
            self._not_empty.notify()

    def get(self, block=True, timeout=None):
        '''
        Remove and return an item from the queue.

        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until an item is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise ('block' is false), return an item if one is immediately
        available, else raise the Empty exception ('timeout' is ignored
        in that case).
        '''
        with self._not_empty:
            if not block:
                if not len(self):
                    raise queue.Empty
            elif timeout is None:
                while not len(self):
                    self._not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time.time() + timeout
                while not len(self):
                    remaining = endtime - time.time()
                    if remaining <= 0.0:
                        raise queue.Empty
                    self._not_empty.wait(remaining)
            return self.queue.popleft()

    def get_nowait(self):
        '''Remove and return an item from the queue without blocking.

        Only get an item if one is immediately available. Otherwise
        raise the Empty exception.
        '''
        return self.get(block=False)

    def size(self):
        return len(self.queue)

    __len__ = size

