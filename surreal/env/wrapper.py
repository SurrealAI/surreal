from .base import Env, ActionType, ObsType
import numpy as np
import surreal.utils as U
from operator import mul
import functools
import sys
import gym
import dm_control
from dm_control.rl.environment import StepType

class SpecFormat(U.StringEnum):
    SURREAL_CLASSIC = ()
    DM_CONTROL = ()
    MUJOCOMANIP = ()


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

    def action_spec(self):
        return self.env.action_spec()

    def observation_spec(self):
        return self.env.observation_spec()

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

class MaxStepWrapper(Wrapper):
    """
        Simple wrapper to limit maximum steps of an environment
    """
    def __init__(self, env, max_steps):
        super().__init__(env)
        if max_steps <= 0:
            raise ValueError('MaxStepWrapper received max_steps')
        self.max_steps = max_steps
        self.current_step = 0

    def _reset(self):
        self.current_step = 0
        return self.env.reset()

    def _step(self, action):
        self.current_step += 1
        observation, reward, done, info = self.env.step(action)
        if self.current_step >= self.max_steps:
            done = True
        return observation, reward, done, info

# putting import inside to allow difference in dependency
class GymAdapter(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, gym.Env)

    def _reset(self):
        obs = self.env.reset()
        return obs, {}

    def observation_spec(self):
        gym_spec = self.env.observation_space
        if isinstance(gym_spec, gym.spaces.Box):
            return {
                'type': 'continuous',
                'dim': gym_spec.shape
            }
        else:
            raise ValueError('Discrete observation currently not supported')
        # TODO: migrate everything to dm_format

    def action_spec(self):
        gym_spec = self.env.action_space
        if isinstance(gym_spec, gym.spaces.Box):
            return {
                'type': 'continuous',
                'dim': gym_spec.shape
            }
        else:
            raise ValueError('Discrete observation currently not supported')
        # TODO: migrate everything to dm_format

    @property
    def spec_format(self):
        return SpecFormat.SURREAL_CLASSIC

class DMControlAdapter(Wrapper):
    def __init__(self, env):
        # dm_control envs don't have metadata
        env.metadata = {}
        super().__init__(env)
        self.screen = None
        assert isinstance(env, dm_control.rl.control.Environment)

    def _step(self, action):
        ts = self.env.step(action)
        reward = ts.reward
        if reward is None:
            # TODO: note that reward is none
            print('None reward')
            reward = 0
        return ts.observation, reward, ts.step_type == StepType.LAST, {}

    def _reset(self):
        ts = self.env.reset()
        return ts.observation, {}

    def _close(self):
        self.env.close()

    @property
    def spec_format(self):
        return SpecFormat.DM_CONTROL

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def _render(self, *args, width=480, height=480, camera_id=1, **kwargs):
        # safe for multiple calls
        import pygame
        pygame.init()
        if not self.screen:
            self.screen = pygame.display.set_mode((width, height))
        else:
            c_width, c_height = self.screen.get_size()
            if c_width != width or c_height != height:
                self.screen = pygame.display.set_mode((width, height))
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        im = self.env.physics.render(width=width, height=height, camera_id=camera_id).transpose((1,0,2))
        pygame.pixelcopy.array_to_surface(self.screen, im)
        pygame.display.update()
        return im

class MujocoManipulationWrapper(Wrapper):
    def __init__(self, env):
        # dm_control envs don't have metadata
        env.metadata = {}
        super().__init__(env)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def _reset(self):
        obs = self.env.reset()
        return obs, {}

    def _close(self):
        self.env.close()

    @property
    def spec_format(self):
        return SpecFormat.DM_CONTROL

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self): # we haven't finalized the action spec of mujocomanip
        # for now I am confirming to dm_control for ease of integration
        low, high = self.env.action_spec()
        return low

    def _render(self, camera_id=0, *args, **kwargs):
        return
        self.env.render(camera_id)
        

def flatten_obs(obs):
    return np.concatenate([v.flatten() for k, v in obs.items()])

class ObservationConcatenationWrapper(Wrapper):
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return flatten_obs(obs), reward, done, info

    def _reset(self):
        obs, info = self.env.reset()
        return flatten_obs(obs), info

    @property
    def spec_format(self):
        return SpecFormat.SURREAL_CLASSIC

    def observation_spec(self):
        return {
            'type': 'continuous',
            'dim': [sum([functools.reduce(mul, list(x.shape) + [1]) for k, x in self.env.observation_spec().items()])],
        }

    def action_spec(self):
        return {
            'type': ActionType.continuous,
            'dim': self.env.action_spec().shape,
        }
    # TODO: what about upper/lower bound information
