"""
Template class for all learners
"""
import surreal.utils as U
from surreal.session import (extend_config,
                     BASE_ENV_CONFIG, BASE_SESSION_CONFIG, BASE_LEARN_CONFIG)
from surreal.distributed import RedisClient, ParameterServer
from tensorplex.loggerplex import LoggerplexClient


class Learner(metaclass=U.AutoInitializeMeta):
    def __init__(self,
                 learn_config,
                 env_config,
                 session_config):
        """
        Write log to self.log

        Args:
            config: a dictionary of hyperparameters. It can include a special
                section "log": {logger configs}
            model: utils.pytorch.Module for the policy network
        """
        self.learn_config = extend_config(learn_config, self.default_config())
        self.env_config = extend_config(env_config, BASE_ENV_CONFIG)
        self.session_config = extend_config(session_config, BASE_SESSION_CONFIG)
        self._client = RedisClient(
            host=self.session_config.ps.host,
            port=self.session_config.ps.port
        )
        self.log = LoggerplexClient(
            client_id='learner',
            host=self.session_config.tensorboard.host,
            port=self.session_config.tensorboard.port
        )

    def _initialize(self):
        """
        For AutoInitializeMeta interface
        TorchBroadcaster deprecated in favor of active pushing

        from surreal.distributed.ps.torch_broadcaster import TorchBroadcaster
        self._broadcaster = TorchBroadcaster(
            redis_client=self._client,
            module_dict=self.module_dict()
        )
        """
        self._parameter_server = ParameterServer(
            redis_client=self._client,
            module_dict=self.module_dict(),
            name=self.session_config.ps.name
        )

    def default_config(self):
        """
        Returns:
            a dict of defaults.
        """
        return BASE_LEARN_CONFIG

    def learn(self, batch_exp):
        """
        Abstract method runs one step of learning

        Args:
            batch_exp: batched experience, can be a tuple of pytorch-ready
                tensor objects (obs_t, obs_t+1, rewards, actions, dones)

        Returns:
            td_error or other values for prioritized replay
        """
        raise NotImplementedError

    def module_dict(self):
        """
        Dict of modules to be broadcasted to the parameter server.
        MUST be consistent with the agent's `module_dict()`
        """
        raise NotImplementedError

    def save(self, file_path):
        """
        Checkpoint to disk
        """
        raise NotImplementedError

    def push_parameters(self, iteration, message=''):
        """
        Learner pushes latest parameters to the parameter server.

        Args:
            iteration: the current number of learning iterations
            message: optional message, must be pickleable.
        """
        self._parameter_server.push(iteration, message=message)
