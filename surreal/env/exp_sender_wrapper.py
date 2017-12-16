from .wrapper import Wrapper
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG, BASE_LEARN_CONFIG
from surreal.distributed.exp_sender import ExpSender, ExpSenderBase
from collections import deque

class ExpSenderWrapper(Wrapper):
    # TODO support N-step obs send
    def __init__(self, env, learner_config, session_config):
        """
        Default sender configs are in BASE_SESSION_CONFIG['sender']
        """
        super().__init__(env)
        config = Config(session_config).extend(BASE_SESSION_CONFIG)
        self.sender = ExpSender(
            host=config.replay.host,
            port=config.replay.port,
            flush_iteration=config.sender.flush_iteration,
        )
        self._obs = None  # obs of the current time step

    def _reset(self):
        self._obs, info = self.env.reset()
        return self._obs, info

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self.sender.send([self._obs, obs_next], action, reward, done, info)
        self._obs = obs_next
        return obs_next, reward, done, info

# Naming may need some change here. 
# The unit of experience is in format state state action reward.
# N-step is supported
class ExpSenderWrapperSSAR(Wrapper):
    def __init__(self, env, learner_config, session_config):
        """
        Default sender configs are in BASE_SESSION_CONFIG['sender']
        """
        super().__init__(env)
        session_config = Config(session_config).extend(BASE_SESSION_CONFIG)
        learner_config = Config(learner_config).extend(BASE_LEARN_CONFIG)
        self.sender = ExpSenderBase(
            host=session_config.replay.host,
            port=session_config.replay.port,
            flush_iteration=session_config.sender.flush_iteration,
        )
        self._obs = None  # obs of the current time step
        self.n_step = learner_config.algo.n_step
        self.gamma = learner_config.algo.gamma
        self.last_n = deque()

    def _reset(self):
        self._obs, info = self.env.reset()
        return self._obs, info

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self.last_n.append([[self._obs, obs_next], action, reward, done, info])
        for i, exp_list in enumerate(self.last_n):
            # Update Next Observation and done to be from the final of n_steps and reward to be weighted sum
            exp_list[0][1] = obs_next
            exp_list[2] += pow(self.gamma, self.n_step - i) * reward
            exp_list[3] = done
        if len(self.last_n) == self.n_step:
            self.send()
        self._obs = obs_next
        return obs_next, reward, done, info

    def send(self):
        obs_array, action, reward, done, info = self.last_n.popleft()
        hash_dict = {
            'obs': obs_array
        }
        nonhash_dict = {
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        }
        self.sender.send(hash_dict, nonhash_dict)