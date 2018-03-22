import sys
import weakref
import surreal.utils as U
import threading
# from .zmq_struct import ZmqQueue
from surreal.distributed.zmq_struct import ZmqPuller
from threading import Thread


class ExperienceCollectorServer(Thread):
    def __init__(self, host, port, exp_handler, load_balanced=True):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.bind = not load_balanced
        self._exp_handler = exp_handler

    def run(self):
        self._weakref_map = weakref.WeakValueDictionary()
        self.puller = ZmqPuller(host=self.host, 
                                port=self.port, 
                                bind=self.bind, 
                                preprocess=U.deserialize)
        while True:
            exp, storage = self.puller.pull()
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
        print('begin retrieve storage')
        print('type(exp) = ', type(exp))
        print('exp = ', exp)
        if isinstance(exp, list):
            for i, e in enumerate(exp):
                exp[i] = self._retrieve_storage(e, storage)
        elif isinstance(exp, dict):
            for key in list(exp.keys()):  # copy keys
                if key.endswith('_hash'):
                    new_key = key[:-len('_hash')]  # delete suffix
                    exp[new_key] = self._retrieve_storage(exp[key], storage)
                    del exp[key]
        elif isinstance(exp, str):
            exphash = exp
            if exphash in self._weakref_map:
                return self._weakref_map[exphash]
            else:
                print('storage[exphash] = ', storage[exphash])
                self._weakref_map[exphash] = storage[exphash]
                return storage[exphash]
        return exp

