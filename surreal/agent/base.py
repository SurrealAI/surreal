"""
A template class that defines base agent APIs
"""
import surreal.utils as U
from surreal.session import Config, BASE_SESSION_CONFIG, extend_config
from surreal.env import BASE_ENV_CONFIG
from surreal.distributed.redis_client import RedisClient
from surreal.distributed.ps.torch_listener import TorchListener


class AgentMode(U.StringEnum):
    training = ()
    eval_stochastic = ()
    eval_deterministic = ()


class Agent(metaclass=U.AutoInitializeMeta):
    def __init__(self,
                 learn_config,
                 env_config,
                 session_config,
                 agent_mode):
        self.learn_config = extend_config(learn_config, self.default_config())
        self.env_config = extend_config(env_config, BASE_ENV_CONFIG)
        self.session_config = extend_config(session_config, BASE_SESSION_CONFIG)
        self.agent_mode = AgentMode[agent_mode]
        self._client = RedisClient(
            host=self.session_config.redis.ps.host,
            port=self.session_config.redis.ps.port
        )

    def _initialize(self):
        # for AutoInitializeMeta interface
        self._listener = TorchListener(
            redis_client=self._client,
            module_dict=self.module_dict()
        )
        self._listener.run_listener_thread()

    def act(self, obs):
        """
        Abstract method for taking actions.
        You should check `self.agent_mode` in the function and change act()
        logic with respect to training VS evaluation.

        Args:
            obs: typically a single obs, make sure to vectorize it first before
                passing to the torch `model`.

        Returns:
            action to be executed in the env
        """
        raise NotImplementedError

    def module_dict(self):
        """
        Returns:
            a dict of name -> surreal.utils.pytorch.Module
        """
        raise NotImplementedError

    def default_config(self):
        """
        Returns:
            a dict of defaults.
        """
        raise NotImplementedError

    def close(self):
        """
        Clean up after the agent exits.
        """
        pass

    def save(self, file_name):
        """
        Checkpoint model to disk
        """
        raise NotImplementedError

    def set_agent_mode(self, agent_mode):
        """
        Args:
            agent_mode: enum of AgentMode class
        """
        self.agent_mode = AgentMode[agent_mode]

