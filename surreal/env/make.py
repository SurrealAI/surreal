import gym
from dm_control import suite
# import mujocomanip
from .wrapper import GymAdapter, DMControlAdapter, ObservationConcatenationWrapper

def make(env_config):
    """
    Makes an environment and populates related fields in env_config
    return env, env_config
    """
    env_name = env_config.env_name
    env_category, env_name = env_name.split(':')
    if env_category == 'gym':
        return make_gym(env_name, env_config)
    elif env_category == 'mujocomanip':
        return make_mujocomanip(env_name, env_config)
    elif env_category == 'dm_control':
        return make_dm_control(env_name, env_config)
    else:
        raise ValueError('Unknown environment category: {}'.format(env_category))

def make_gym(env_name, env_config):
    env = gym.make(env_name)
    env = GymAdapter(env)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config

def make_mujocomanip(env_name, env_config):
    raise NotImplementedError()
    pass

def make_dm_control(env_name, env_config):
    domain_name, task_name = env_name.split('-')
    env = suite.load(domain_name=domain_name, task_name=task_name)
    env = DMControlAdapter(env)
    env = ObservationConcatenationWrapper(env)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config

