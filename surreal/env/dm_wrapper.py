import numpy as np
import sys
import collections
import dm_control
from dm_control.suite.wrappers import pixels
from dm_control.rl.environment import StepType
from .wrapper import Wrapper, SpecFormat
from .base import ActionType


class DMControlDummyWrapper(Wrapper):
    
    '''
    Dummy wrapper for deepmind control environment using pixels.  The output of
    observation_spec and action_spec will match the output for a dm_control environment
    using pixels.Wrapper().  This is used by the learner to get the action and observation
    specs without initializing a pixels wrapper.
    '''

    def __init__(self, env):
        # dm_control envs don't have metadata
        env.metadata = {}
        super().__init__(env)

    @property
    def spec_format(self):
        return SpecFormat.DM_CONTROL

    def observation_spec(self):
        modality = collections.OrderedDict([('pixels', dm_control.rl.specs.ArraySpec(shape=(84, 84, 3), dtype=np.dtype('float32')))])
        return modality

    def action_spec(self):
        return self.env.action_spec()

class DMControlAdapter(Wrapper):
    def __init__(self, env, is_pixel_input):
        # dm_control envs don't have metadata
        env.metadata = {}
        super().__init__(env)
        self.screen = None
        self.is_pixel_input = is_pixel_input
        assert (isinstance(env, dm_control.rl.control.Environment) or
            isinstance(env, pixels.Wrapper) or
            isinstance(env, DMControlDummyWrapper))

    def _add_modality(self, obs):
        if self.is_pixel_input:
            renamed_obs = collections.OrderedDict([('camera0', obs['pixels'])])
            return collections.OrderedDict([('pixel', renamed_obs)])
        else:
            return collections.OrderedDict([('low_dim', obs)])

    def _step(self, action):
        ts = self.env.step(action)
        reward = ts.reward
        if reward is None:
            # TODO: note that reward is none
            print('None reward')
            reward = 0
        return self._add_modality(ts.observation), reward, ts.step_type == StepType.LAST, {}

    def _reset(self):
        ts = self.env.reset()
        return self._add_modality(ts.observation), {}

    def _close(self):
        self.env.close()

    @property
    def spec_format(self):
        return SpecFormat.SpecFormat.DM_CONTROL

    def observation_spec(self):
        obs_spec = collections.OrderedDict()
        for modality, v in self._add_modality(self.env.observation_spec()).items():
            modality_spec = collections.OrderedDict()
            for key, obs_shape in v.items():
                # Deepmind observation spec uses Deepmind data types, we just need shape as tuple
                modality_spec[key] = obs_shape.shape
            obs_spec[modality] = modality_spec
        return obs_spec

    def action_spec(self):
        return {
            'type': ActionType.continuous,
            'dim': self.env.action_spec().shape, # DM_control returns int, we want all dim to be tuple
        }

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

        im = self.env.physics.render(width=width,
            height=height, camera_id=camera_id).transpose((1,0,2))
        pygame.pixelcopy.array_to_surface(self.screen, im)
        pygame.display.update()
        return im