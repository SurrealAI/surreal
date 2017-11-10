"""
A template class that defines base agent APIs
"""
import threading
import surreal.utils as U

# TODO torch_listener inside __init__
# add a .module_list() class for torch listener
# In __init__


class AgentMode(U.StringEnum):
    training = ()
    eval_stochastic = ()
    eval_deterministic = ()


class Agent(object):
    def __init__(self, config, agent_mode):
        U.assert_type(config, dict)
        self.agent_mode = AgentMode.get_enum(agent_mode)

    def act(self, obs, *args, **kwargs):
        """
        Abstract method for taking actions.
        You should check `self.agent_mode` in the function and change act()
        logic with respect to training VS evaluation.

        Args:
            obs: typically a single obs, make sure to vectorize it first before
                passing to the torch `model`.
            *args, **kwargs: other info to make the action, such as the current
                epsilon exploration value.

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
        self.agent_mode = AgentMode.get_enum(agent_mode)

