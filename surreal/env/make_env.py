from surreal.env.video_env import VideoWrapper
from .wrapper import GymAdapter, DMControlAdapter, ObservationConcatenationWrapper

def make_env(env_config):
    """
    Makes an environment and populates related fields in env_config
    return env, env_config
    """
    env_name = env_config.env_name
    env_category, env_name = env_name.split(':')
    record_video = env_config.video.record_video
    env, env_config = None, None
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
    domain_name, task_name = env_name.split('-')
    env = suite.load(domain_name=domain_name, task_name=task_name)
    env = DMControlAdapter(env)
    env = ObservationConcatenationWrapper(env)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config

