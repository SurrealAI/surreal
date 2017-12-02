import surreal.utils as U
from .config import Config
from .default_configs import BASE_SESSION_CONFIG
from tensorplex import TensorplexClient, LoggerplexClient


# Referenced by run_tensorplex_server.py for "normal_group" and "numbered_group"
AGENT_GROUP_NAME = 'agents'
EVAL_GROUP_NAME = 'eval'
STATS_GROUP_NAME = 'stats'


class Loggerplex(LoggerplexClient):
    def __init__(self, name, session_config):
        C = Config(session_config).extend(BASE_SESSION_CONFIG)
        super().__init__(
            name,
            host=C.loggerplex.host,
            port=C.loggerplex.port
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


class EvalTensorplex(TensorplexClient):
    def __init__(self, eval_id, session_config):
        U.assert_type(eval_id, str)
        C = Config(session_config).extend(BASE_SESSION_CONFIG)
        super().__init__(
            '{}/{}'.format(EVAL_GROUP_NAME, eval_id),
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
            '{}/{}'.format(STATS_GROUP_NAME, section_name),
            host=C.tensorplex.host,
            port=C.tensorplex.port
        )
