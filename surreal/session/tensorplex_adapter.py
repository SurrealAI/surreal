import surreal.utils as U
from tensorplex import TensorplexClient, LoggerplexClient


__all__ = ['Loggerplex', 'AgentTensorplex', 'StatsTensorplex']


# Referenced by run_tensorplex_server.py for "normal_group" and "numbered_group"
AGENT_GROUP_NAME = 'agents'
NONAGENT_GROUP_NAME = 'stats'


class Loggerplex(LoggerplexClient):
    def __init__(self, name, session_config):
        super().__init__(
            name,
            host=session_config['tensorplex']['host'],
            port=session_config['tensorplex']['port'],
        )


class AgentTensorplex(TensorplexClient):
    def __init__(self, agent_id, session_config):
        U.assert_type(agent_id, int)
        super().__init__(
            '{}/{}'.format(AGENT_GROUP_NAME, agent_id),
            host=session_config['tensorplex']['host'],
            port=session_config['tensorplex']['port'],
        )


class StatsTensorplex(TensorplexClient):
    def __init__(self, section_name, session_config):
        """
        Args:
            section_name: will show up on Tensorboard as a separate section
        """
        super().__init__(
            '{}/{}'.format(NONAGENT_GROUP_NAME, section_name),
            host=session_config['tensorplex']['host'],
            port=session_config['tensorplex']['port'],
        )
