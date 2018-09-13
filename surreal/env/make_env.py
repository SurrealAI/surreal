import os
from multiprocessing import Process, Queue
from surreal.env.video_env import VideoWrapper
from .wrapper import GymAdapter
from .wrapper import FrameStackWrapper, GrayscaleWrapper, TransposeWrapper, FilterWrapper
from .wrapper import ObservationConcatenationWrapper, MujocoManipulationWrapper


def make_env_config(env_config, mode=None):
    """
    Forks a process, creates the environment and generate the config
    This makes sure that when we initializes an environment using
    make_env, we have not created and then deleted another one (just
    to get the dimension of input). Many rendering related things
    can break when created and destroyed.

    e.g. If you create a mujoco_py MjOffscreenRenderContext,
    delete it, fork the process and re-create the context, you
    will get a setfault

    Args:
        env_config: see make_env
        mode: see make_env
    """
    q = Queue()
    p = Process(target=_make_env_wrapped, args=(q, env_config, mode))
    print("HI8")
    p.start()
    print("HI9")
    config = q.get()
    print("HI0")
    p.join()
    print("DONE")
    return config


def _make_env_wrapped(q, env_config, mode):
    """
    For running make env in another process
    """
    print("HI")
    _, config = make_env(env_config, mode)
    print("HI4")
    q.put(config)
    print(type(config))
    print("HI5")


def make_env(env_config, mode=None):
    """
    Makes an environment and populates related fields in env_config
    return env, env_config
    Args:
        overrides(str): point to the override in env_config to 
                        provide differnt kwargs to the initializer of the environment
    """
    env_name = env_config.env_name
    env_category, env_name = env_name.split(':')
    if mode == 'eval' and 'eval_mode' in env_config:
        for k, v in env_config.eval_mode.items():
            env_config[k] = v
    if env_category == 'gym':
        env, env_config = make_gym(env_name, env_config)
    elif env_category == 'mujocomanip':
        env, env_config = make_mujocomanip(env_name, env_config)
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


def make_mujocomanip(env_name, env_config):
    import MujocoManip
    
    demo_config = None if env_config.demonstration is None or \
                  not env_config.demonstration.use_demo else env_config.demonstration

    env = MujocoManip.make(
        env_name,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=env_config.pixel_input,
        camera_height=84,
        camera_width=84,
        render_collision_mesh=False,
        render_visual_mesh=True,
        camera_name='tabletop',
        use_object_obs=(not env_config.pixel_input),
        camera_depth=env_config.use_depth,
        reward_shaping=True,
        demo_config=env_config.demonstration,
    )
    env = MujocoManipulationWrapper(env, env_config)
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

