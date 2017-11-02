import gym


def env_wrapper_by_name(env, classname):
    """Given an a gym environment possibly wrapped multiple times, returns a wrapper
    of class named classname or raises ValueError if no such wrapper was applied

    Parameters
    ----------
    env: gym.Env of gym.Wrapper
        gym environment
    classname: str
        name of the wrapper

    Returns
    -------
    wrapper: gym.Wrapper
        wrapper named classname
    """
    currentenv = env
    while True:
        if classname == currentenv.class_name():
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)
