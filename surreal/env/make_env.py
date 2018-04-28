from surreal.env.video_env import VideoWrapper
from .wrapper import GymAdapter, DMControlAdapter, DMControlDummyWrapper
from .wrapper import FrameStackWrapper, GrayscaleWrapper, TransposeWrapper, FilterWrapper
from .wrapper import ObservationConcatenationWrapper, MujocoManipulationWrapper
from dm_control.suite.wrappers import pixels
import os


def make_env(env_config, session_config, learner_config, eval_mode=False):
    """
    Makes an environment and populates related fields in env_config
    return env, env_config
    """
    env_name = env_config.env_name
    env_category, env_name = env_name.split(':')
    # Video recording should only be done in eval agents
    record_video = env_config.video.record_video and eval_mode
    if record_video and os.getenv('DISABLE_MUJOCO_RENDERING'):
        # Record_video is set to true by default but tmux on mac won't always have rendering
        record_video = False
        print('Won\'t produce videos because rendering is turned off on this machine')
    if env_category == 'gym':
        env, env_config = make_gym(env_name, env_config)
    elif env_category == 'mujocomanip':
        env, env_config = make_mujocomanip(env_name, env_config)
    elif env_category == 'dm_control':
        env, env_config = make_dm_control(env_name, env_config, learner_config, record_video)
    else:
        raise ValueError('Unknown environment category: {}'.format(env_category))
    if record_video:
        env = VideoWrapper(env, env_config, session_config)
    return env, env_config


def make_gym(env_name, env_config):
    import gym
    env = gym.make(env_name)
    env = GymAdapter(env)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config


def make_mujocomanip(env_name, env_config, learner_config):
    import MujocoManip
    env = MujocoManip.make(env_name, use_camera_obs=False, horizon=50000) 
    env = MujocoManipulationWrapper(env)
    env = ObservationConcatenationWrapper(env, learner_config)
    # set to true and to receive camera input
    # Remove observation concatenation wrapper and parse observation spec properly
    env_config.action_spec = env.observation_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config


def make_dm_control(env_name, env_config, learner_config, record_video=False):
    from dm_control import suite
    pixel_input = env_config.pixel_input
    domain_name, task_name = env_name.split('-')
    env = suite.load(domain_name=domain_name, task_name=task_name)
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
        
    # TODO: what to do with reward visualization
    # Reward visualization should only be done in the eval agent
    # env = suite.load(domain_name=domain_name, task_name=task_name, visualize_reward=record_video)

    env = DMControlAdapter(env, pixel_input)
    env = FilterWrapper(env, learner_config)
    env = ObservationConcatenationWrapper(env, learner_config)
    if pixel_input:
        env = TransposeWrapper(env, learner_config)
        env = GrayscaleWrapper(env, learner_config)
        env = FrameStackWrapper(env, env_config, learner_config)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config

