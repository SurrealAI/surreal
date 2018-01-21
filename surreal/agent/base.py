"""
A template class that defines base agent APIs
"""
import surreal.utils as U
from surreal.session import (
    Loggerplex, AgentTensorplex, EvalTensorplex,
    PeriodicTracker, PeriodicTensorplex, extend_config,
    BASE_ENV_CONFIG, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG
)
from surreal.distributed import ParameterClient


class AgentMode(U.StringEnum):
    training = ()
    eval_stochastic = ()
    eval_deterministic = ()


class AgentCore(metaclass=U.AutoInitializeMeta):
    def __init__(self, ps_host, ps_port, agent_mode):
        """
        Write all logs to self.log
        """
        self.agent_mode = AgentMode[agent_mode]
        self._ps_client = None
        self._ps_host = ps_host
        self._ps_port = ps_port

    def _initialize(self):
        "implements AutoInitializeMeta meta class."
        self._ps_client = ParameterClient(
            host=self._ps_host,
            port=self._ps_port,
            module_dict=self.module_dict()
        )

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

    def fetch_parameter(self):
        """
        Update agent by pulling parameters from parameter server.
        """
        return self._ps_client.fetch_parameter()

    def fetch_parameter_info(self):
        """
        Update agent by pulling parameters from parameter server.
        """
        return self._ps_client.fetch_info()

    def set_agent_mode(self, agent_mode):
        """
        Args:
            agent_mode: enum of AgentMode class
        """
        self.agent_mode = AgentMode[agent_mode]



class Agent(AgentCore):
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 agent_id,
                 agent_mode):
        """
        Write all logs to self.log
        """
        self.learner_config = extend_config(learner_config, self.default_config())
        self.env_config = extend_config(env_config, BASE_ENV_CONFIG)
        self.session_config = extend_config(session_config, BASE_SESSION_CONFIG)
        super().__init__(
            ps_host=self.session_config.ps.host,
            ps_port=self.session_config.ps.port,
            agent_mode=agent_mode,
        )
        if self.agent_mode == AgentMode.training:
            U.assert_type(agent_id, int)
            logger_name = 'agent-{}'.format(agent_id)
            self.tensorplex = AgentTensorplex(
                agent_id=agent_id,
                session_config=self.session_config
            )
        else:
            logger_name = 'eval-{}'.format(agent_id)
            self.tensorplex = EvalTensorplex(
                eval_id=str(agent_id),
                session_config=self.session_config
            )
        self._periodic_tensorplex = PeriodicTensorplex(
            tensorplex=self.tensorplex,
            period=self.session_config.tensorplex.update_schedule.agent,
            is_average=True,
            keep_full_history=False
        )
        self.log = Loggerplex(
            name=logger_name,
            session_config=self.session_config
        )

    def update_tensorplex(self, tag_value_dict, global_step=None):
        self._periodic_tensorplex.update(tag_value_dict, global_step)

    def default_config(self):
        """
        Returns:
            a dict of defaults.
        """
        return BASE_LEARNER_CONFIG

    def reset(self):
        pass