"""
Learner: pushes parameters to key "ps" and
    param info to hashmap key "psinfo" on Redis.
Agent: pulls parameters from key "ps"
Evaluator: pulls param info from "psinfo" and do diagnostics.
"""
import pickle
import time
import surreal.utils as U
from surreal.distributed.zmq_struct import ZmqPub, ZmqReq, ZmqSimpleServer, ZmqSubClient, ZmqTimeoutError
from surreal.distributed.proxy import ZmqLoadBalancerThread
from surreal.distributed.module_dict import ModuleDict
from threading import Lock
from multiprocessing import Process
import os


class ParameterPublisher(object):
    """
    Learner side
    """
    def __init__(self, port, module_dict):
        """
        Args:
            name: key that points to the parameter binary on Redis.
                "<name>info" will be the key to the info Redis hashmap.
                e.g. "psinfo" -> {'time': 32541.6, 'iteration': 1200}
        """
        self._publisher = ZmqPub(
            host='*',
            port=port,
            preprocess=U.serialize,
        )
        if not isinstance(module_dict, ModuleDict):
            module_dict = ModuleDict(module_dict)
        self._module_dict = module_dict

    def publish(self, iteration, message=''):
        """
        Called by learner.

        Args:
            iteration: current learning iteration
            message: any pickleable data
        """
        binary = self._module_dict.dumps()
        info = {
            'time': time.time(),
            'iteration': iteration,
            'message': message,
            'hash': U.binary_hash(binary)
        }
        self._publisher.pub(topic='ps', data=(binary, info))


class ShardedParameterServer(object):
    def __init__(self, config):
        self.ps_config = config
        self.shards = self.ps_config.shards

        self.frontend_port = os.environ['SYMPH_PS_FRONTEND_PORT']
        self.backend_port = os.environ['SYMPH_PS_BACKEND_PORT']

        self.parameter_serving_frontend_add = "tcp://*:{}".format(self.frontend_port)
        self.parameter_serving_backend_add = "tcp://*:{}".format(self.backend_port)

        self.proxy = None
        self.workers = []

    def launch(self):
        self.proxy = ZmqLoadBalancerThread(in_add=self.parameter_serving_frontend_add,
                                           out_add=self.parameter_serving_backend_add,
                                           pattern='router-dealer')

        self.proxy.start()

        self.workers = []

        publish_host = os.environ['SYMPH_PARAMETER_PUBLISH_HOST']
        publish_port = os.environ['SYMPH_PARAMETER_PUBLISH_PORT']
        for i in range(self.shards):
            worker = ParameterServer(
                publish_host=publish_host,
                publish_port=publish_port,
                serving_host='localhost',
                serving_port=self.backend_port,
                load_balanced=True,
            )
            worker.start()
            self.workers.append(worker)
            # break

    def join(self):
        for i, worker in enumerate(self.workers):
            worker.join()
            U.report_exitcode(worker.exitcode, 'replay-{}'.format(i))
        self.proxy.join()

class ParameterServer(Process):
    # TODO support multiple PS
    """
    Standalone script for PS node that runs in an infinite loop.
    PS subscribes to upstream (learner) and REPs to downstream (agent)
    """
    def __init__(self,
                 publish_host,
                 publish_port,
                 serving_host,
                 serving_port,
                 load_balanced=False):
        """

        Args:
            publish_host: learner side publisher server
            publish_port:
            agent_port: PS server that responds to agent fetch_parameter requests
        """
        Process.__init__(self)
        self.publish_host = publish_host
        self.publish_port = publish_port
        self.serving_host = serving_host
        self.serving_port = serving_port
        # self.serving_port = 7005
        self.load_balanced = load_balanced
        # storage
        self.parameters = None
        self.param_info = None

    def run(self):
        self._subscriber = ZmqSubClient(
            host=self.publish_host,
            port=self.publish_port,
            handler=self._set_storage,
            topic='ps',
            preprocess=U.deserialize,
        )
        self._server = ZmqSimpleServer(
            host=self.serving_host,
            port=self.serving_port,
            handler=self._handle_agent_request,
            preprocess=U.deserialize,
            postprocess=U.serialize,
            load_balanced=self.load_balanced,
        )
        self._subscriber.start()
        self._server.start()
        print('Parameter server started')
        self._subscriber.join()
        self._server.join()
        # print('Finished')
        # return 'abc'

    def _set_storage(self, data):
        self.parameters, self.param_info = data

    def _handle_agent_request(self, request):
        """
        Reply to agents pulling params

        Args:
            request: 3 types
             - "info": only info
             - "parameter:<last_hash>": returns None if hash is not changed
                since the last request
             - "both:<last_hash>": returns (None, info) if hash is not
                changed, otherwise (param, info)
        """
        if request == 'info':
            return self.param_info
        elif request.startswith('parameter'):
            if self.parameters is None:
                return None, ''
            _, last_hash = request.split(':', 1)
            current_hash = self.param_info['hash']
            if last_hash == current_hash:  # param not changed
                return None, current_hash
            else:
                return self.parameters, current_hash
        elif request.startswith('both'):
            if self.parameters is None:
                return None, None
            _, last_hash = request.split(':', 1)
            if last_hash == self.param_info['hash']:  # param not changed
                return None, self.param_info
            else:
                return self.parameters, self.param_info
        else:
            raise ValueError('invalid request: '+str(request))



class ParameterClient(object):
    """
    Agent side
    """
    def __init__(self, host, port, timeout=2):
        """
        Args:
            host: parameter server host
            port:
            module_dict:
        """
        self.host = host
        self.port = port
        self._last_hash = ''
        self.alive = False
        self.timeout = timeout

    def fetch_parameter(self):
        """
        Called by agent. Pulls from PS ONLY WHEN the parameter hash changes to
        prevent duplicate fetching. No-op when duplicate.

        Returns:
            True if parameter is actually fetched (changed since last request).
        """
        print('fp-1')
        client = ZmqReq(
            host=self.host,
            port=self.port,
            preprocess=U.serialize,
            postprocess=U.deserialize,
            timeout=self.timeout
        )
        print('fp-2')
        try:
            response = client.request('parameter:' + self._last_hash)
        except ZmqTimeoutError:
            self.report_fetch_parameter_failed()
            return False
        print('fp-3')
        self.report_fetch_parameter_success()
        print('fp-4')
        param, cur_hash = response
        self._last_hash = cur_hash
        if param:
            return param
        else:
            return False

    def fetch_parameter_with_info(self):
        """
        Called by agent. Pulls from PS ONLY WHEN the parameter hash changes to
        prevent duplicate fetching. No-op when duplicate.

        Returns:
            (info dict, True if parameter is actually fetched)
        """
        print('fpi-1')
        client = ZmqReq(
            host=self.host,
            port=self.port,
            preprocess=U.serialize,
            postprocess=U.deserialize,
            timeout=self.timeout
        )
        print('fpi-2')
        try:
            response = client.request('both:' + self._last_hash)
        except ZmqTimeoutError:
            self.report_fetch_parameter_failed()
            return False, {}
        print('fpi-3')
        self.report_fetch_parameter_success()
        param, info = response
        self._last_hash = info['hash'] if info else ''
        print('fpi-4')
        if param:
            return param, info
        else:
            return False, info

    def fetch_info(self):
        return self._client.request('info')

    def report_fetch_parameter_failed(self):
        if self.alive:
            self.alive = False
            print('Parameter client request timed out')

    def report_fetch_parameter_success(self):
        if not self.alive:
            self.alive = True
            print('Parameter client came back alive')



