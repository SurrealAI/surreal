"""
A template class that defines base environment APIs
"""
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

    def _seed(self, seed=None):
        return []

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

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return self._seed(seed)

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


class Wrapper(Env):
    # Clear metadata so by default we don't override any keys.
    metadata = {}
    # Make sure self.env is always defined, even if things break early.
    env = None

    def __init__(self, env):
        self.env = env
        # Merge with the base metadata
        metadata = self.metadata
        self.metadata = self.env.metadata.copy()
        self.metadata.update(metadata)
        self._ensure_no_double_wrap()

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _ensure_no_double_wrap(self):
        env = self.env
        while True:
            if isinstance(env, Wrapper):
                if env.class_name() == self.class_name():
                    raise RuntimeError(
                        "Attempted to double wrap with Wrapper: {}"
                        .format(self.__class__.__name__)
                    )
                env = env.env
            else:
                break

    def _step(self, action):
        return self.env.step(action)

    def _reset(self):
        return self.env.reset()

    def _render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def _close(self):
        return self.env.close()

    def _seed(self, seed=None):
        return self.env.seed(seed)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped


class ObservationWrapper(Wrapper):
    def _reset(self):
        observation = self.env.reset()
        return self._observation(observation)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return self._observation(observation)

    def _observation(self, observation):
        raise NotImplementedError


class RewardWrapper(Wrapper):
    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        return self._reward(reward)

    def _reward(self, reward):
        raise NotImplementedError


class ActionWrapper(Wrapper):
    def _step(self, action):
        action = self.action(action)
        return self.env.step(action)

    def action(self, action):
        return self._action(action)

    def _action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        return self._reverse_action(action)

    def _reverse_action(self, action):
        raise NotImplementedError
