from kingkong.utils import *
from collections import namedtuple
from .atari_wrappers_deprecated import wrap_dqn


class ObsToTensor(gym.ObservationWrapper):
    def _observation(self, obs):
        return torch.FloatTensor(obs)


class ActionUnTensor(gym.ActionWrapper):
    def _action(self, action):
        return TC.to_scalar(action)


# when you make an EpisodicMonitor, make sure you do it on the _same_ env instance
EnvTuple = namedtuple('EnvTuple', ['env', 'monitor_env', 'action_dim'])


def cartpole_env(return_tuple=True):
    env = gym.make('CartPole-v0')
    env = U.EpisodeMonitor(ActionUnTensor(env))
    if return_tuple:
        return EnvTuple(env=env,
                        monitor_env=env,
                        action_dim=env.action_space.n)
    else:
        return env


def atari_env(name, return_tuple=True):
    """
    DO NOT convert observation to tensor in the wrapper for DQN!!!
    FrameStack wrapper uses LazyFrame, which will preserve replay memory
    DO NOT use ScaledFloatFrame, or it will undo the memory optimization
    atari frames are stored in np.array(uint8), which takes less memory than float
    Only convert to tensor when you actually need to compute.
    """
    name = U.atari_name_cap(name) + 'NoFrameskip-v4'
    env = gym.make(name)
    if return_tuple:
        monitor_env = U.EpisodeMonitor(env)
        return EnvTuple(env=wrap_dqn(monitor_env),
                        monitor_env=monitor_env,
                        action_dim=monitor_env.action_space.n)
    else:
        return wrap_dqn(env)

