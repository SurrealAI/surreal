"""
A template class that defines base agent APIs
"""
import threading
import surreal.utils as U


class Agent(object):
    def __init__(self, model, action_mode):
        U.assert_type(model, U.Module)
        self.model = model
        self.action_mode = action_mode
        self._forward_lock = threading.Lock()

    def _act(self, obs, model, action_mode, *args, **kwargs):
        """
        Abstract method for taking actions.

        Args:
            obs: typically a single obs, make sure to vectorize it first before
                passing to the torch `model`.
            action_mode: see `set_action_mode`
            *args, **kwargs: other info to make the action, such as the current
                epsilon exploration value.

        Returns:
            action
        """
        raise NotImplementedError

    def initialize(self, *args, **kwargs):
        """
        Initialize before interaction with the env
        """
        pass

    def close(self):
        """
        Clean up after the agent exits.
        """
        pass

    def act(self, obs, *args, **kwargs):
        """
        Wraps around self._act() abstract method.
        Uses thread lock to ensure that forward-prop and parameter updating
        cannot happen at the same time.

        Args:
            obs:
            **kwargs: passed to self._act()
        """
        with self._forward_lock:
            return self._act(obs, self.model, self.action_mode, *args, **kwargs)

    def get_lock(self):
        """
        Called by parameter server listener thread to avoid race condition
        with forward-prop.
        """
        return self._forward_lock

    def get_model(self):
        return self.model

    def set_action_mode(self, action_mode):
        """
        Args:
            action_mode: example modes
            - "train"
            - "eval-d" for deterministic evaluation
            - "eval-s" for stochastic evaluation
        """
        self.action_mode = action_mode

    def save(self, file_name):
        with self._forward_lock:
            self.model.save(file_name)

