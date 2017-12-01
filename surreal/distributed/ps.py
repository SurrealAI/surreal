"""
Learner: pushes parameters to key "ps" and
    param info to hashmap key "psinfo" on Redis.
Agent: pulls parameters from key "ps"
Evaluator: pulls param info from "psinfo" and do diagnostics.
"""
import pickle
import time
import surreal.utils as U
from .module_dict import ModuleDict
from .zmq_struct import (ZmqPublishServer, ZmqSubscribeClient,
                         ZmqClient, ZmqServer)


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
        self._publisher = ZmqPublishServer(
            port=port,
            is_pyobj=True
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
        self._publisher.publish((binary, info), topic='ps')


class ParameterServer(object):
    # TODO support multiple PS
    """
    Standalone script for PS node that runs in an infinite loop.
    PS subscribes to upstream (learner) and REPs to downstream (agent)
    """
    def __init__(self,
                 learner_host,
                 learner_port,
                 agent_port):
        self._subscriber = ZmqSubscribeClient(
            host=learner_host,
            port=learner_port,
            handler=self._set_storage,
            topic='ps',
            is_pyobj=True
        )
        self._server = ZmqServer(
            port=agent_port,
            handler=self._handle_agent_request,
            is_pyobj=True
        )
        # storage
        self.parameters = None
        self.param_info = None

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
                return None, None
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

    def run_loop(self):
        "blocking"
        self._subscriber.run_loop(block=False)
        self._server.run_loop(block=True)


class ParameterClient(object):
    """
    Agent side
    """
    def __init__(self, host, port, module_dict):
        """
        Args:
            host: parameter server host
            port:
            module_dict:
        """
        self._client = ZmqClient(
            host=host,
            port=port,
            is_pyobj=True
        )
        if not isinstance(module_dict, ModuleDict):
            module_dict = ModuleDict(module_dict)
        self._module_dict = module_dict
        self._last_hash = ''

    def fetch_parameter(self):
        """
        Called by agent. Pulls from PS ONLY WHEN the parameter hash changes to
        prevent duplicate fetching. No-op when duplicate.

        Returns:
            True if parameter is actually fetched (changed since last request).
        """
        param, cur_hash = self._client.request('parameter:' + self._last_hash)
        self._last_hash = cur_hash
        if param:
            self._module_dict.loads(param)
            return True
        else:
            return False

    def fetch_parameter_with_info(self):
        """
        Called by agent. Pulls from PS ONLY WHEN the parameter hash changes to
        prevent duplicate fetching. No-op when duplicate.

        Returns:
            (info dict, True if parameter is actually fetched)
        """
        param, info = self._client.request('both:' + self._last_hash)
        self._last_hash = info['hash'] if info else ''
        if param:
            self._module_dict.loads(param)
            return True, info
        else:
            return False, info

    def fetch_info(self):
        return self._client.request('info')
