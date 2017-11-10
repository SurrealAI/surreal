"""
Template class for all learners
"""
import surreal.utils as U
from easydict import EasyDict
from surreal.distributed.redis_client import RedisClient
from surreal.distributed.ps.torch_broadcaster import TorchBroadcaster


class Learner(metaclass=U.AutoInitializeMeta):
    def __init__(self, config):
        """

        Args:
            config: a dictionary of hyperparameters. It can include a special
                section "log": {logger configs}
            model: utils.pytorch.Module for the policy network
        """
        U.assert_type(config, dict)
        self.config = C = EasyDict(config)
        self._client = RedisClient(
            host=C.redis.ps.host,
            port=C.redis.ps.port
        )
        # TODO better logging
        log_kwargs = C.log if 'log' in C else {}
        self.log = U.Logger.get_logger('Learner', **log_kwargs)

    def _initialize(self):
        # for AutoInitializeMeta interface
        self._broadcaster = TorchBroadcaster(
            redis_client=self._client,
            module_dict=self.module_dict()
        )

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
