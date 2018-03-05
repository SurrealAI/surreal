import queue
import zmq
from threading import Thread, Lock
import surreal.utils as U
from tensorplex import Logger


zmq_logger = Logger.get_logger(
    'zmq',
    stream='stdout',
    time_format='hms',
    show_level=True,
)

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

##
# Sender receiver implemented by REQ-REP
##

class ZmqSender(ZmqSocketWrapper):
    def __init__(self, host, port, serializer=None):
        super().__init__(host=host, port=port, mode=zmq.REQ, bind=False)
        self.serializer = serializer
        self.establish()

    def send(self, data):
        if self.serializer:
            data = self.serializer(data)
        self.socket.send(data)
        resp = self.socket.recv()
        return resp

class ZmqReceiver(ZmqSocketWrapper):
    def __init__(self, host, port, bind=True, deserializer=None):
        super().__init__(host=host, port=port, mode=zmq.REP, bind=bind)
        self.deserializer = deserializer
        self.establish()

    def recv(self):
        data = self.socket.recv()
        if self.deserializer:
            data = self.deserializer(data)
        self.socket.send(b'ack') # doesn't matter
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


class ZmqReq(ZmqSocketWrapper):
    def __init__(self, host, port, bind=False, preprocess=None, postprocess=None):
        super().__init__(host=host, port=port, mode=zmq.REQ, bind=bind)
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.establish()

    def request(self,data):
        if self.preprocess:
            data = self.preprocess(data)
        self.socket.send(data)
        rep = self.socket.recv()
        if self.postprocess:
            rep = self.postprocess(rep)
        return rep


class ZmqPub(ZmqSocketWrapper):
    def __init__(self, host, port, hwm=1, preprocess=None):
        super().__init__(host=host, port=port, mode=zmq.PUB, bind=True)
        self.socket.set_hwm(hwm)
        self.preprocess = preprocess
        self.establish()

    def pub(self, topic, data):
        topic = U.str2bytes(topic)
        if self.preprocess:
            data = self.preprocess(data)
        self.socket.send_multipart([topic, data])


class ZmqSub(ZmqSocketWrapper):
    def __init__(self, host, port, topic, hwm=1, deserializer=None):
        super().__init__(host=host, port=port, mode=zmq.SUB, bind=False)
        topic = U.str2bytes(topic)
        self.socket.set_hwm(hwm)
        self.socket.setsockopt(zmq.SUBSCRIBE, topic)
        self.establish()

    def recv(self):
        _, data = self.socket.recv_multipart()
        return data


class ZmqSubClient(Thread):
    def __init__(self, host, port, topic, handler, preprocess=None, hwm=1):
        Thread.__init__(self)
        self.sub = ZmqSub(host, port, topic, hwm)
        self.host = host
        self.port = port
        self.topic = topic
        self.preprocess = preprocess
        self.handler = handler

    def run(self):
        zmq_logger.infofmt('SubClient listening for topic {} on {}:{}', 
                             self.topic, self.host, self.port)
        while True:
            data = self.sub.recv()
            if self.preprocess:
                data = self.preprocess(data)
            self.handler(data)


class ZmqAsyncServerWorker(Thread):
    """
    replay -> learner, replay handling learner's requests
    """
    def __init__(self, context, handler, preprocess, postprocess):
        Thread.__init__(self)
        self.context = context
        self._handler = handler

    def run(self):
        socket = self.context.socket(zmq.REP)
        socket.connect('inproc://worker')
        while True:
            req = socket.recv()
            if self.preprocess:
                req = self.preprocess(req)
            res = self._handler(req)
            if self.postprocess:
                res = self.postprocess(res)
            socket.send(res)
        socket.close()


class ZmqAsyncServer(Thread):
    """
    replay -> learner, manages ZmqServerWorker pool
    Async REQ-REP server
    """
    def __init__(self, host, port, handler, num_workers=1,
                    load_balanced=False, preprocess=None, postprocess=None):
        """
        Args:
            port:
            handler: takes the request (pyobj) and sends the response
        """
        Thread.__init__(self)
        self.port = port
        self.host = host
        self.handler = handler
        self.num_workers = num_workers
        self.load_balanced = load_balanced
        self.preprocess = preprocess
        self.postprocess = postprocess

    def run(self):
        context = zmq.Context()
        router_sw = ZmqSocketWrapper(mode=zmq.ROUTER, 
                                  bind=(not self.load_balanced),
                                  host=self.host,
                                  port=self.port,
                                  context=context)
        router = router_sw.establish()

        dealer_sw = ZmqSocketWrapper(mode=zmq.ROUTER, 
                                  bind=True,
                                  address="inproc://worker", 
                                  context=context)
        dealer = dealer_sw.establish()

        workers = []
        for worker_id in range(self.num_workers):
            worker = ZmqAsyncServerWorker(context, self.handler, self.preprocess, self.postprocess)
            worker.start()
            workers.append(worker)

        # http://zguide.zeromq.org/py:mtserver
        # http://api.zeromq.org/3-2:zmq-proxy
        # **WARNING**: zmq.proxy() must be called AFTER the threads start,
        # otherwise the program hangs!
        # Before calling zmq_proxy() you must set any socket options, and
        # connect or bind both frontend and backend sockets.
        zmq.proxy(router, dealer)
        # should never reach
        router.close()
        dealer.close()
        context.term()


class ZmqSimpleServer(Thread):
    def __init__(self, host, port, handler,
                 load_balanced=False, 
                 preprocess=None, 
                 postprocess=None):
        Thread.__init__(self)
        self.sw = ZmqSocketWrapper(mode=zmq.REP,host=host,port=port,bind=(not load_balanced))
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.handler = handler

    def run(self):
        socket = self.sw.establish()
        while True:
            req = socket.recv()
            if self.preprocess:
                req = self.preprocess(req)
            res = self.handler(req)
            if self.postprocess:
                res = self.postprocess(res)
            socket.send(res)
