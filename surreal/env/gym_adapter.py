import gym
from .base import Wrapper


class GymAdapter(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, gym.Env)

    def _reset(self):
        obs = self.env.reset()
        return obs, {}
