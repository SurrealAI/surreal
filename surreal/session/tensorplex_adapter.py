import surreal.utils as U
from .config import Config
from .default_configs import BASE_SESSION_CONFIG
from tensorplex import TensorplexClient, LoggerplexClient


__all__ = ['Loggerplex', 'AgentTensorplex', 'StatsTensorplex']


# Referenced by run_tensorplex_server.py for "normal_group" and "numbered_group"
AGENT_GROUP_NAME = 'agents'
NONAGENT_GROUP_NAME = 'stats'


class Loggerplex(LoggerplexClient):
    def __init__(self, name, session_config):
        C = Config(session_config).extend(BASE_SESSION_CONFIG)
        super().__init__(
            name,
            host=C.tensorplex.host,
            port=C.tensorplex.port
        )


class AgentTensorplex(TensorplexClient):
    def __init__(self, agent_id, session_config):
        U.assert_type(agent_id, int)
        C = Config(session_config).extend(BASE_SESSION_CONFIG)
        super().__init__(
            '{}/{}'.format(AGENT_GROUP_NAME, agent_id),
            host=C.tensorplex.host,
            port=C.tensorplex.port
        )


class StatsTensorplex(TensorplexClient):
    def __init__(self, section_name, session_config):
        """
        Args:
            section_name: will show up on Tensorboard as a separate section
        """
        C = Config(session_config).extend(BASE_SESSION_CONFIG)
        super().__init__(
            '{}/{}'.format(NONAGENT_GROUP_NAME, section_name),
            host=C.tensorplex.host,
            port=C.tensorplex.port
        )
