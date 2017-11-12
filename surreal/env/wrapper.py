from .base import Env


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
        # self.obs_spec = env.obs_spec
        # self.action_spec = env.action_spec

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

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped


class ObsWrapper(Wrapper):
    def _reset(self):
        observation, info = self.env.reset()
        return self._observation(observation), info

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



