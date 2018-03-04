from surreal.env.video_env import VideoWrapper
from .wrapper import GymAdapter, DMControlAdapter, ObservationConcatenationWrapper, DMControlDummyWrapper
from dm_control.suite.wrappers import pixels
import os

def make_env(env_config):
    """
    Makes an environment and populates related fields in env_config
    return env, env_config
    """
    env_name = env_config.env_name
    env_category, env_name = env_name.split(':')
    record_video = env_config.video.record_video
    if env_category == 'gym':
        env, env_config = make_gym(env_name, env_config)
    elif env_category == 'mujocomanip':
        env, env_config = make_mujocomanip(env_name, env_config)
    elif env_category == 'dm_control':
        env, env_config = make_dm_control(env_name, env_config)
    else:
        raise ValueError('Unknown environment category: {}'.format(env_category))
    if record_video:
        env = VideoWrapper(env, env_config)
    return env, env_config


def make_gym(env_name, env_config):
    import gym
    env = gym.make(env_name)
    env = GymAdapter(env)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config


def make_mujocomanip(env_name, env_config):
    # import mujocomanip
    raise NotImplementedError()
    pass


def make_dm_control(env_name, env_config):
    from dm_control import suite
    pixel_input = env_config.pixel_input
    domain_name, task_name = env_name.split('-')
    env = suite.load(domain_name=domain_name, task_name=task_name)
    print(env.action_spec())
    print(env.observation_spec())
    if pixel_input:
        if os.getenv('DISABLE_MUJOCO_RENDERING'):
            # We are asking for rendering on a pod that cannot support rendering, 
            # This happens in GPU based learners when we only want to create the environment
            # to see the dimensions.
            # So we will add a dummy environment
            # TODO: add a dummy wrapper that only contains the correct specs
            env = DMControlDummyWrapper(env) #...
        else:
            env = pixels.Wrapper(env, render_kwargs={'height': 84, 'width': 84, 'camera_id': 0})
            print(env.action_spec())
            print(env.observation_spec())
        # TODO: add our custom frame stacking wrapper
            
        
    env = DMControlAdapter(env)
    print(env.action_spec())
    print(env.observation_spec())
    env = ObservationConcatenationWrapper(env)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    print('done make')
    return env, env_config

