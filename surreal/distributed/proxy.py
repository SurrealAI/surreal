import zmq
from multiprocessing import Process
from threading import Thread

class ZmqLoadBalancer(object):
    def __init__(self, in_add, out_add, context=None, pattern='router-dealer'):
        """
            Receive requests from in_add and forward to out_add
        Args:
            @in_add: address that frontend binds to
            @out_add: address that backend binds to
            @context: Provided context for zmq, if None, a private context is created
            @patter: 'router-dealer' or 'pull-push', what zmq proxy pattern to use
        """
        self.initialize()
        self.in_add = in_add
        self.out_add = out_add
        self.context = context
        self.pattern = pattern
        if pattern == 'router-dealer':
            self.frontend_protocol = zmq.ROUTER
            self.backend_protocol = zmq.DEALER
        elif pattern == 'pull-push':
            self.frontend_protocol = zmq.PULL
            self.backend_protocol = zmq.PUSH
        else:
            raise ValueError('Unkown zmq proxy patter {}. \
                Please choose router-dealer or pull-push'.format(pattern))
        if self.context is None:
            self.context = zmq.Context()
            
    def run(self):
        frontend = self.context.socket(self.frontend_protocol)
        frontend.bind(self.in_add)

        # Socket facing services
        backend  = self.context.socket(self.backend_protocol)
        backend.bind(self.out_add)

        print('Forwarding traffic from {} to {}. Pattern: {}'.format(self.in_add,
                                                                 self.out_add,
                                                                 self.pattern))
        zmq.proxy(frontend, backend)

        # We never get here…
        frontend.close()
        backend.close()
        self.context.term()

    def initialize(self):
        pass


class ZmqLoadBalancerThread(ZmqLoadBalancer, Thread):
    def initialize(self):
        Thread.__init__(self)

class ZmqLoadBalancerProcess(ZmqLoadBalancer, Process):
    def initialize(self):
        Process.__init__(self)
        



