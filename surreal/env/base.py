"""
A template class that defines base environment APIs
"""
import surreal.utils as U


class ActionType(U.StringEnum):
    continuous = ()
    discrete = ()


class ObsType(U.StringEnum):
    TODO = ()


class _EnvMeta(type):  # DEPRECATED
    """
    Ensure that env always has `action_spec` and `obs_spec` after __init__
    """
    def __call__(self, *args, **kwargs):
        env = super().__call__(*args, **kwargs)
        # env must have instance vars action_spec and obs_spec
        # they must be dict
        assert hasattr(env, 'action_spec') and isinstance(env.action_spec, dict)
        assert hasattr(env, 'obs_spec') and isinstance(env.obs_spec, dict)
        return env


class Env(object):
    """The main Env class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    When implementing an environment, override the following methods
    in your subclass:

        _step
        _reset
        _render
        _close
        _seed

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    metadata = {}

    def __new__(cls, *args, **kwargs):
        # We use __new__ since we want the env author to be able to
        # override __init__ without remembering to call super.
        env = super(Env, cls).__new__(cls)
        env._closed = False
        return env

    # Override in SOME subclasses
    def _close(self):
        pass

    # Override in ALL subclasses
    def _step(self, action):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _render(self, *args, **kwargs):
        pass

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        return self._step(action)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
            info (dict)
        """
        return self._reset()

    def render(self, *args, **kwargs):
        """Renders the environment.
        """
        return self._render(*args, **kwargs)

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.

        Also close the rendering threads, if any.
        """
        # _closed will be missing if this instance is still
        # initializing.
        if not hasattr(self, '_closed') or self._closed:
            return
        self._close()
        # If an error occurs before this line, it's possible to
        # end up with double close.
        self._closed = True

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{}>'.format(type(self).__name__)
