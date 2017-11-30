import queue
import collections
import threading


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
