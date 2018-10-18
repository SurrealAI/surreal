import weakref
from threading import Thread
import surreal.utils as U
from caraml.zmq import ZmqReceiver


class ExperienceCollectorServer(Thread):
    """
        Accepts experience from agents,
        deduplicates experience whenever possible
    """
    def __init__(self, host, port, exp_handler, load_balanced=True):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.load_balanced = load_balanced
        self._exp_handler = exp_handler
        # To be initialized in run()
        self._weakref_map = None
        self.receiver = None

    def run(self):
        """
            Starts the server loop
        """
        self._weakref_map = weakref.WeakValueDictionary()
        self.receiver = ZmqReceiver(host=self.host,
                                    port=self.port,
                                    bind=not self.load_balanced,
                                    deserializer=U.deserialize)
        while True:
            exp, storage = self.receiver.recv()
            experience_list = self._retrieve_storage(exp, storage)
            for exp in experience_list:
                self._exp_handler(exp)

    def _retrieve_storage(self, exp, storage):
        """
        Args:
            exp: a nested dict or list
                Only dict keys that end with `_hash` will be retrieved.
                The processed key will see `_hash` removed
            storage: chunk of storage sent with the exps
        """
        if isinstance(exp, list):
            for i, e in enumerate(exp):
                exp[i] = self._retrieve_storage(e, storage)

        elif isinstance(exp, dict):
            for key in list(exp.keys()):  # copy keys
                if key.endswith('_hash'):
                    new_key = key[:-len('_hash')]  # delete suffix
                    exp[new_key] = self._retrieve_storage(exp[key], storage)
                    del exp[key]
                else:
                    exp[key] = self._retrieve_storage(exp[key], storage)

        elif isinstance(exp, str):
            exphash = exp
            if exphash in self._weakref_map:
                return self._weakref_map[exphash]
            else:
                self._weakref_map[exphash] = storage[exphash]
                return storage[exphash]
        return exp
