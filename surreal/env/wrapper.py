from .base import Env, ActionType, ObsType
import numpy as np
import surreal.utils as U
import collections
from collections import deque
from operator import mul
import functools
import sys
import gym


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
        self._obsspec = None
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

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def _step(self, action):
        return self.env.step(action)

    def _reset(self):
        obs, info = self.env.reset()
        self._assert_conforms_to_spec(obs)
        return obs

    def _render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def _close(self):
        return self.env.close()

    def _assert_conforms_to_spec(self, obs):
        if not self._obsspec:
            self._obsspec = self.observation_spec()
        for modality in self._obsspec:
            for key in self._obsspec[modality]:
                assert self._obsspec[modality][key] == obs[modality][key].shape

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
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env


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
    def __init__(self, env, env_config):
        super().__init__(env)
        assert not env_config.pixel_input, "Pixel input training not supported with OpenAI Gym"
        assert isinstance(env, gym.Env)
        self.env = env

    def _add_modality(self, obs):
        obs = {
            'flat_inputs': obs
        }
        return collections.OrderedDict([('low_dim', obs)])

    def _reset(self):
        obs = self.env.reset()
        return self._add_modality(obs), {}

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._add_modality(obs)
        return obs, reward, done, info

    def observation_spec(self):
        gym_spec = self.env.observation_space
        if isinstance(gym_spec, gym.spaces.Box):
            return self._add_modality(gym_spec.shape)
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

    def _close(self):
        self.env.close()

    def _render(self):
        return self.env.render(mode='rgb_array')

    @property
    def spec_format(self):
        return SpecFormat.SURREAL_CLASSIC


class RobosuiteWrapper(Wrapper):
    def __init__(self, env, env_config):
        # dm_control envs don't have metadata
        env.metadata = {}
        super().__init__(env)
        self.use_depth = env_config.use_depth and env_config.pixel_input
        self._input_list = env_config.observation
        self._action_repeat = env_config.action_repeat or 1

    def _add_modality(self, obs, verbose=False):
        pixel_modality = collections.OrderedDict()
        flat_modality = collections.OrderedDict()
        for key in obs:
            if key == 'image' and 'camera0' in self._input_list['pixel']:
                pixel_modality['camera0'] = obs[key]
            elif key in self._input_list['pixel']:
                pixel_modality[key] = obs[key]
            elif key in self._input_list['low_dim']:
                flat_modality[key] = obs[key]
            elif verbose:
                print('Mujoco: skipping observation key:', key)
        obs = collections.OrderedDict()
        if len(pixel_modality) > 0:
            obs['pixel'] = pixel_modality
        if len(flat_modality) > 0:
            obs['low_dim'] = flat_modality
        return obs

    def _step(self, action):
        rewards = []
        for repeat in range(self._action_repeat):
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break
        reward = np.mean(rewards)

        if self.use_depth:
            obs['image'] = np.concatenate((obs['image'], np.expand_dims(obs['depth'], 2)), 2)

        return self._add_modality(obs), reward, done, info

    def _reset(self):
        obs = self.env.reset()

        if self.use_depth:
            obs['image'] = np.concatenate((obs['image'], np.expand_dims(obs['depth'], 2)), 2)
        
        return self._add_modality(obs), {}

    def _close(self):
        self.env.close()

    @property
    def spec_format(self):
        return SpecFormat.MUJOCOMANIP

    def observation_spec(self):
        spec = self.env.observation_spec()

        if self.use_depth:
            spec['image'] = np.concatenate((spec['image'], np.expand_dims(spec['depth'], 2)), 2)

        for k in spec:
            spec[k] = tuple(np.array(spec[k]).shape)

        return self._add_modality(spec, verbose=True)

    def action_spec(self): # we haven't finalized the action spec of mujocomanip
        return {'dim': (self.env.dof,), 'type': 'continuous'}

    def _render(self, *args, **kwargs):
        return self.env.sim.render(camera_name='frontview',
                                   height=512,
                                   width=512,
                                   depth=False)


class ObservationConcatenationWrapper(Wrapper):
    def __init__(self, env, concatenated_obs_name='flat_inputs'):
        super().__init__(env)
        self._concatenated_obs_name = concatenated_obs_name

    def _flatten_obs(self, obs):
        flat_observations = []
        if 'low_dim' in obs:
            for modality, v in obs['low_dim'].items():
                flat_observations.append(v)
            if len(flat_observations) > 0:
                flat_observations = np.concatenate(flat_observations)
                del obs['low_dim']
                obs['low_dim'] = collections.OrderedDict([(self._concatenated_obs_name, flat_observations)])
        return obs

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._flatten_obs(obs), reward, done, info

    def _reset(self):
        obs, info = self.env.reset()
        return self._flatten_obs(obs), info

    @property
    def spec_format(self):
        return SpecFormat.SURREAL_CLASSIC

    def observation_spec(self):
        spec = self.env.observation_spec()
        flat_observation_dim = 0
        if 'low_dim' in spec:
            for k, shape in spec['low_dim'].items():
                assert len(shape) == 1
                flat_observation_dim += shape[0]
            spec['low_dim'] = collections.OrderedDict([(self._concatenated_obs_name, (flat_observation_dim,))])
        return spec

    def action_spec(self):
        return self.env.action_spec()


class TransposeWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _transpose(self, obs):
        if 'pixel' in obs:
            for key in obs['pixel']:
                # input is (H, W, 3), we want (C, H, W) == (3, 84, 84)
                obs['pixel'][key] = obs['pixel'][key].transpose((2, 0, 1))
        return obs

    def _reset(self):
        obs, info = self.env.reset()
        return self._transpose(obs), info

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._transpose(obs), reward, done, info

    def observation_spec(self):
        spec = self.env.observation_spec()
        if 'pixel' in spec:
            for key in spec['pixel']:
                H, W, C = spec['pixel'][key]
                # We transpose to (C, H, W) to work with pytorch convolutions
                visual_dim = (C, H, W)
                spec['pixel'][key] = visual_dim
        return spec


class GrayscaleWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _grayscale(self, obs):
        for key in obs['pixel']:
            observation_modality = obs['pixel'][key]
            C, H, W = observation_modality.shape
            # For now, we expect an RGB image
            assert C == 3
            obs['pixel'][key] = np.mean(observation_modality, 0, 'uint8').reshape(1, H, W)
        return obs

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._grayscale(obs)
        return obs, reward, done, info

    def _reset(self):
        obs, info = self.env.reset()
        scaled = self._grayscale(obs)
        return scaled, info

    @property
    def spec_format(self):
        return SpecFormat.SURREAL_CLASSIC

    def observation_spec(self):
        spec = self.env.observation_spec()
        # TODO: make constant pixel_modality='pixel'
        for key in spec['pixel']:
            dimensions = spec['pixel'][key]
            C, H, W = dimensions
            # We expect rgb for now
            assert C == 3
            spec['pixel'][key] = (1, H, W)
        return spec

    def action_spec(self):
        return self.env.action_spec()

class FrameStackWrapper(Wrapper):
    def __init__(self, env, env_config):
        super().__init__(env)
        self.n = env_config.frame_stacks
        self.frame_stack_concatenate_on_env = env_config.frame_stack_concatenate_on_env
        self._history = deque(maxlen=self.n)

    def _stacked_observation(self, obs):
        '''
        Assumes self._history contains the last n frames from the environment
        Concatenates the frames together along the depth axis
        '''
        #stacked_obs_dict_unordered = collections.OrderedDict()
        #stacked_obs_dict_unordered['pixel'] = collections.OrderedDict()
        new_pixel_modality = collections.OrderedDict()

        for key in obs['pixel']:
            obs_stacked = []
            for history_obs in self._history:
                obs_stacked.append(history_obs['pixel'][key])
            assert len(obs_stacked) <= self.n
            if self.frame_stack_concatenate_on_env:
                stacked = np.concatenate(obs_stacked, axis=0)
            else:
                stacked = obs_stacked
            #stacked_obs_dict_unordered['pixel'][key] = stacked
            new_pixel_modality[key] = stacked
        next_stacked_dict = collections.OrderedDict()
        for key in obs:
            if key == 'pixel':
                next_stacked_dict[key] = new_pixel_modality
            else:
                next_stacked_dict[key] = obs[key]
        return next_stacked_dict

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self._history.append(obs_next)
        obs_next_stacked = self._stacked_observation(obs_next)
        return obs_next_stacked, reward, done, info

    def _reset(self):
        obs, info = self.env.reset()
        for i in range(self.n):
            self._history.append(obs)
        a = self._stacked_observation(obs)
        return a, info

    @property
    def spec_format(self):
        return SpecFormat.SURREAL_CLASSIC

    def observation_spec(self):
        spec = self.env.observation_spec()
        if 'pixel' in spec:
            for key in spec['pixel']:
                dimensions = spec['pixel'][key]
                C, H, W = dimensions
                if C > 4:
                    print('WARNING: Received input of size (C, H, W) == ', dimensions)
                    print('number of channels is greater than 4')
                spec['pixel'][key] = (C * self.n, H, W)
        return spec

    def action_spec(self):
        return self.env.action_spec()

class FilterWrapper(Wrapper):
    '''
    Given the inputs allowed in env_config.observation, reject any inputs
    not specified in the config.
    '''
    def __init__(self, env, env_config):
        super().__init__(env)
        self._allowed_items = env_config.observation

    def _filtered_obs(self, obs, verbose=False):
        filtered = collections.OrderedDict()
        for modality in obs:
            if modality in self._allowed_items:
                modality_spec = collections.OrderedDict()
                for key in obs[modality]:
                    if key in self._allowed_items[modality]:
                        modality_spec[key] = obs[modality][key]
                    elif verbose:
                        print('Skipping observation key:', modality, '/', key)
                filtered[modality] = modality_spec
        return filtered

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        return self._filtered_obs(obs_next), reward, done, info

    def _reset(self):
        obs, info = self.env.reset()
        return self._filtered_obs(obs), info

    @property
    def spec_format(self):
        return SpecFormat.SURREAL_CLASSIC

    def observation_spec(self):
        spec = self.env.observation_spec()
        return self._filtered_obs(spec, verbose=True)

    def action_spec(self):
        return self.env.action_spec()
