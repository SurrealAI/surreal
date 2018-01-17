from .wrapper import Wrapper
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG
from surreal.distributed.exp_sender import ExpSender
from collections import deque


exp_sender_wrapper_registry = {}

def register_exp_sender_wrapper(target_class):
    exp_sender_wrapper_registry[target_class.__name__] = target_class

def expSenderWrapperFactory(env, learner_config, session_config):
    session_config = Config(session_config).extend(BASE_SESSION_CONFIG)
    learner_config = Config(learner_config).extend(BASE_LEARNER_CONFIG)
    return exp_sender_wrapper_registry[learner_config.algo.experience](env, learner_config, session_config)

class ExpSenderWrapperMeta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        register_exp_sender_wrapper(cls)
        return cls

# https://effectivepython.com/2015/02/02/register-class-existence-with-metaclasses/
class ExpSenderWrapperBase(Wrapper, metaclass=ExpSenderWrapperMeta):
    def __init__(self, env, learner_config, session_config):
        """
        Default sender configs are in BASE_SESSION_CONFIG['sender']
        """
        super().__init__(env)
        # TODO: initialize config in a unified place 
        self.session_config = Config(session_config).extend(BASE_SESSION_CONFIG)
        self.learner_config = Config(learner_config).extend(BASE_LEARNER_CONFIG)
        self.sender = ExpSender(
            host=self.session_config.replay.host,
            port=self.session_config.replay.port,
            flush_iteration=self.session_config.sender.flush_iteration,
        )
        

class ExpSenderWrapperSSAR(ExpSenderWrapperBase):
    def __init__(self, env, learner_config, session_config):
        super.__init__(env, learner_config, session_config)
        self._obs = None  # obs of the current time step

    def _reset(self):
        self._obs, info = self.env.reset()
        return self._obs, info

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self.send([[self._obs, obs_next], action, reward, done, info])
        self._obs = obs_next
        return obs_next, reward, done, info

    def send(self, data):
        obs_array, action, reward, done, info = data
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

# Naming may need some change here. 
# The unit of experience is in format state state action reward.
# N-step is supported
class ExpSenderWrapperSSARNStep(ExpSenderWrapperSSAR):
    def __init__(self, env, learner_config, session_config):
        """
        Default sender configs are in BASE_SESSION_CONFIG['sender']
        """
        super().__init__(env, learner_config, session_config)
        self.n_step = self.learner_config.algo.n_step
        self.gamma = self.learner_config.algo.gamma
        self.last_n = deque()

    def _reset(self):
        self._obs, info = self.env.reset()
        self.last_n.clear()
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
            self.send(self.last_n.popleft())
        self._obs = obs_next
        return obs_next, reward, done, info

class ExpSenderWrapperMultiStep(ExpSenderWrapperBase):
    def send(self, data):
        action_arr, reward_arr, done_arr, info_arr = [], [], [], []
        hash_dict = {}
        nonhash_dict = {}
        for index, (obs, action, reward, done, info) in enumerate(data):
            # Store observations in a deduplicated way
            hash_dict[str(index)] = obs
            action_arr.append(action)
            reward_arr.append(reward)
            done_arr.append(done)
            info_arr.append(info)
        nonhash_dict = {
            'action_arr': action_arr,
            'reward_arr': reward_arr,
            'done_arr': done_arr,
            'info_arr': info_arr,
            'n_step': len(data),
        }
        self.sender.send(hash_dict, nonhash_dict)


class ExpSenderWrapperStackN(ExpSenderWrapperMultiStep):
    """
    Sends obs * n, action * n, reward * n, done, info
    """
    def __init__(self, env, learner_config, session_config):
        super().__init__(env, learner_config, session_config)
        self._obs = None  # obs of the current time step
        self.n_step = self.learner_config.algo.n_step
        self.last_n = deque()

    def _reset(self):
        self._obs, info = self.env.reset()
        self.last_n.clear()
        return self._obs, info

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self.last_n.append([self._obs, action, reward, done, info])
        if len(self.last_n) == self.n_step:
            self.send(self.last_n)
            # TODO: allow configuring behavior here
            self.last_n.popleft()
        return obs_next, reward, done, info


class ExpSenderWrapperTrajectory(ExpSenderWrapperMultiStep):
    """
    Sends obs * n, action * n, reward * n, done, info
    """
    def __init__(self, env, learner_config, session_config):
        super().__init__(env, learner_config, session_config)
        self._obs = None  # obs of the current time step
        self.trajectory = deque()

    def _reset(self):
        self._obs, info = self.env.reset()
        self.trajectory.clear()
        return self._obs, info

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self.trajectory.append([self._obs, action, reward, done, info])
        if done:
            self.send(self.trajectory)
            self.trajectory.clear()
        return obs_next, reward, done, info
