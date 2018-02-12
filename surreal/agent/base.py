"""
A template class that defines base agent APIs
"""
import surreal.utils as U
from surreal.session import (
    Loggerplex, AgentTensorplex, EvalTensorplex,
    PeriodicTracker, PeriodicTensorplex, extend_config,
    BASE_ENV_CONFIG, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG
)
from surreal.distributed import ParameterClient, ModuleDict
import time


class AgentMode(U.StringEnum):
    training = ()
    eval_stochastic = ()
    eval_deterministic = ()

agent_registry = {}

def register_agent(target_class):
    agent_registry[target_class.__name__] = target_class

def agentFactory(agent_name):
    return agent_registry[agent_name]

class AgentMeta(U.AutoInitializeMeta):
    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        register_agent(cls)
        return cls

class AgentCore(metaclass=AgentMeta):
    def __init__(self, ps_host, ps_port, agent_mode):
        """
        Write all logs to self.log
        """
        self.agent_mode = AgentMode[agent_mode]
        self._ps_client = None
        self._ps_host = ps_host
        self._ps_port = ps_port

    def _initialize(self):
        """
            implements AutoInitializeMeta meta class.
        """
        self._ps_client = ParameterClient(
            host=self._ps_host,
            port=self._ps_port,
            module_dict=self.module_dict(),
        )

    def pre_action(self, obs):
        """
            Called before act is called by agent main script
        """
        pass

    def post_action(self, obs, action, obs_next, reward, done, info):
        """
            Called before act is called by agent main script.
            TODO: move experience generation to here so that agent has control over it.
        """
        pass

    def pre_episode(self):
        """
            Called by agent process.
            Can beused to reset internal states before an episode starts
        """
        pass

    def post_episode(self):
        """
            Called by agent process.
            Can beused to reset internal states after an episode ends
            I.e. after the post_action when done = True
        """
        pass

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
        params, info = self._ps_client.fetch_parameter_with_info()
        return params, info

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
    """
        Important: When extending this class, make sure to follow the init method signature so that 
        orchestrating functions can properly initialize custom agents.

        TODO: Extend the initilization to allow custom non-config per-agent settings.
            To be used to have a heterogeneous agent population
    """
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

        # Parameter update related logging
        self.last_parameter_time = None
        # record how long the current parameter have been used
        self.actions_per_param_update = 0
        self.episodes_per_param_update = 0

    def update_tensorplex(self, tag_value_dict, global_step=None):
        self._periodic_tensorplex.update(tag_value_dict, global_step)

    def default_config(self):
        """
        Returns:
            a dict of defaults.
        """
        return BASE_LEARNER_CONFIG

    def fetch_parameter(self):
        """
            Extends base class fetch_parameters to add some logging
        """
        params, info = super().fetch_parameter()
        if params:
            self.on_parameter_fetched(params, info)

    def on_parameter_fetched(self, params, info):
        """
            Method called when a new parameter is fetched. Free to be inherited by subclasses.
        """
        # The time it takes for parameter to go from learner to agent
        if self.agent_mode == AgentMode.training:
            delay = time.time() - info['time']
            self.update_tensorplex({'parameter_publish_delay': delay,
                                    'actions_per_param_update': self.actions_per_param_update,
                                    'episodes_per_param_update': self.episodes_per_param_update
                                    })
            self.actions_per_param_update = 0
            self.episodes_per_param_update = 0


    def post_action(self, obs, action, obs_next, reward, done, info):
        super().post_action(obs, action, obs_next, reward, done, info)
        if self.agent_mode == AgentMode.training:
            self.actions_per_param_update += 1
            if done:
                self.episodes_per_param_update += 1