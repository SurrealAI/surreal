import os
import gc
from surreal.env.video_env import VideoWrapper
import surreal.utils as U
from .wrapper import (
    GymAdapter,
    FrameStackWrapper,
    GrayscaleWrapper,
    TransposeWrapper,
    FilterWrapper,
    ObservationConcatenationWrapper,
    RobosuiteWrapper
    )


def make_env_config(env_config, mode=None):
    """
    Issue: Surreal-tmux gives segfault on running robosuite environments.
    Cause: On launching a surreal process, an instance of RL environment is
        created to obtain observation and action dimension info. This
        environment can hold reference to openGL context that is not safe
        for multiprocessing. The environment is dereferenced but not
        necessarily garbage collected before fork/exec happens,
        causing memory issues.
    Solution: make_env_config method deallocates the environment and explicitly
    calls the garbage collector. Use this method to obtain observation and
        action dimensions.

    Same as make_env, but explicitly deallocates the environment to avoid
    further complications

    Returns:
        Populated env_config
    """
    env, env_config = make_env(env_config, mode)
    del env
    gc.collect()
    return env_config


def make_env(env_config, mode=None):
    """
    Makes an environment and populates related fields in env_config
    return env, env_config
    Args:
        mode(str): allow differnt kwargs to the initializer of the environment
    """
    env_name = env_config.env_name
    env_category, env_name = env_name.split(':')
    if mode == 'eval' and 'eval_mode' in env_config:
        for k, v in env_config.eval_mode.items():
            env_config[k] = v
    if env_category == 'gym':
        env, env_config = make_gym(env_name, env_config)
    elif env_category == 'robosuite':
        env, env_config = make_robosuite(env_name, env_config)
    elif env_category == 'dm_control':
        env, env_config = make_dm_control(env_name, env_config)
    else:
        raise ValueError('Unknown environment category: {}'.format(env_category))
    return env, env_config


def make_gym(env_name, env_config):
    import gym
    env = gym.make(env_name)
    env = GymAdapter(env, env_config)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config


def make_robosuite(env_name, env_config):
    import robosuite

    env = robosuite.make(
        env_name,
        has_renderer=env_config.render,
        ignore_done=True,
        use_camera_obs=env_config.pixel_input,
        has_offscreen_renderer=env_config.pixel_input,
        camera_height=84,
        camera_width=84,
        render_collision_mesh=False,
        render_visual_mesh=True,
        camera_name='agentview',
        use_object_obs=(not env_config.pixel_input),
        camera_depth=env_config.use_depth,
        reward_shaping=True,
        # demo_config=env_config.demonstration,
    )
    env = RobosuiteWrapper(env, env_config)
    env = FilterWrapper(env, env_config)
    env = ObservationConcatenationWrapper(env)
    if env_config.pixel_input:
        env = TransposeWrapper(env)
        if env_config.use_grayscale:
            env = GrayscaleWrapper(env)
        if env_config.frame_stacks:
            env = FrameStackWrapper(env, env_config)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config


def make_dm_control(env_name, env_config):
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
    from .dm_wrapper import DMControlAdapter, DMControlDummyWrapper
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
    env = FilterWrapper(env, env_config)
    env = ObservationConcatenationWrapper(env)
    if pixel_input:
        env = TransposeWrapper(env)
        env = GrayscaleWrapper(env)
        if env_config.frame_stacks > 1:
            env = FrameStackWrapper(env, env_config)
    env_config.action_spec = env.action_spec()
    env_config.obs_spec = env.observation_spec()
    return env, env_config
