"""
Template class for all learners
"""
import surreal.utils as U
from surreal.session import Config, BASE_SESSION_CONFIG, extend_config
from surreal.env import BASE_ENV_CONFIG
from surreal.distributed.redis_client import RedisClient
from surreal.distributed.ps.torch_broadcaster import TorchBroadcaster


class Learner(metaclass=U.AutoInitializeMeta):
    def __init__(self,
                 learn_config,
                 env_config,
                 session_config):
        """

        Args:
            config: a dictionary of hyperparameters. It can include a special
                section "log": {logger configs}
            model: utils.pytorch.Module for the policy network
        """
        self.learn_config = extend_config(learn_config, self.default_config())
        self.env_config = extend_config(env_config, BASE_ENV_CONFIG)
        self.session_config = extend_config(session_config, BASE_SESSION_CONFIG)
        self._client = RedisClient(
            host=self.session_config.redis.ps.host,
            port=self.session_config.redis.ps.port
        )
        # TODO better logging
        # log_kwargs = C.log if 'log' in C else {}
        # self.log = U.Logger.get_logger('Learner', **log_kwargs)

    def _initialize(self):
        # for AutoInitializeMeta interface
        self._broadcaster = TorchBroadcaster(
            redis_client=self._client,
            module_dict=self.module_dict()
        )

    def default_config(self):
        """
        Returns:
            a dict of defaults.
        """
        raise NotImplementedError

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

    def save(self, folder):
        """
        Checkpoint to disk
        """
        raise NotImplementedError

    def broadcast(self, message=''):
        self._broadcaster.broadcast(message=message)
