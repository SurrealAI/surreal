import threading


class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        self._stop_event = threading.Event() # stoppable thread
        super().__init__(*args, **kwargs)

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()
