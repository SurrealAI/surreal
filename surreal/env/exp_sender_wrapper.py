from .wrapper import Wrapper
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG, ConfigError
from surreal.distributed.exp_sender import ExpSender
from collections import deque
import resource
exp_sender_wrapper_registry = {}

def register_exp_sender_wrapper(target_class):
    exp_sender_wrapper_registry[target_class.__name__] = target_class

def expSenderWrapperFactory(name):
    return exp_sender_wrapper_registry[name]
    
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
        They contain communication level information
        
        Algorithm specific experience generation parameters should live in learner_config
        """
        super().__init__(env)
        # TODO: initialize config in a unified place 
        self.session_config = Config(session_config).extend(BASE_SESSION_CONFIG)
        self.learner_config = Config(learner_config).extend(BASE_LEARNER_CONFIG)
        self.sender = ExpSender(
            host=self.session_config.replay.collector_frontend_host,
            port=self.session_config.replay.collector_frontend_port,
            flush_iteration=self.session_config.sender.flush_iteration,
        )
        

class ExpSenderWrapperSSAR(ExpSenderWrapperBase):
    """
        Sends experience in format
        {   
            'obs': [state, next state]
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        }
    """
    def __init__(self, env, learner_config, session_config):
        super().__init__(env, learner_config, session_config)
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
class ExpSenderWrapperSSARNStepBootstrap(ExpSenderWrapperSSAR):
    """
        Sends observations in format
        {   
            'obs': [state, next_state]
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        }
        but next_state is n_steps after state and reward is cumulated reward over n_steps
        Used for n_step reward computations.

        Requires:
            @self.learner_config.algo.n_step: number of steps to cumulate over
            @self.learner_config.algo.gamma: discount factor
    """
    def __init__(self, env, learner_config, session_config):
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
        for i, exp_list in enumerate(self.last_n):
            # Update Next Observation and done to be from the final of n_steps and reward to be weighted sum
            exp_list[0][1] = obs_next
            exp_list[2] += pow(self.gamma, self.n_step - i - 1) * reward
            exp_list[3] = done
        self.last_n.append([[self._obs, obs_next], action, reward, done, info])
        if len(self.last_n) == self.n_step:
            self.send(self.last_n.popleft())

        self._obs = obs_next
        return obs_next, reward, done, info

class ExpSenderWrapperMultiStep(ExpSenderWrapperBase):
    """
        Base class for all classes that send experience in format
        {   
            'obs_arr': [state_1, ..., state_n]
            'obs_next': [state_{n + 1}]
            'action_arr': [action_1, ...],
            'reward_arr': [reward_1, ...],
            'done_arr': [done_1, ...],
            'info_arr': [info_1, ...],
            'n_step': n, length of all arrays,
        }
    """
    def send(self, data, obs_next):
        obs_arr, action_arr, reward_arr, done_arr, info_arr = [], [], [], [], []
        hash_dict = {}
        nonhash_dict = {}
        for index, (obs, action, reward, done, info) in enumerate(data):
            # Store observations in a deduplicated way
            obs_arr.append(obs)
            action_arr.append(action)
            reward_arr.append(reward)
            done_arr.append(done)
            info_arr.append(info)

        hash_dict = {
            'obs_arr': obs_arr,
            'obs_next': obs_next,
        }
        nonhash_dict = {
            'action_arr': action_arr,
            'reward_arr': reward_arr,
            'done_arr': done_arr,
            'info_arr': info_arr,
            'n_step': len(data),
        }
        self.sender.send(hash_dict, nonhash_dict)


class ExpSenderWrapperMultiStepMovingWindowWithInfo(ExpSenderWrapperBase):
    """
        Base class for all classes that send experience in format
        {   
            'obs': [state_1, ..., state_n]
            'obs_next': [state_{n + 1}]
            'actions': [action_1, ...],
            'rewards': [reward_1, ...],
            'dones': [done_1, ...],
            'persistent_infos': [infolist_1, ...]
            'onetime_infos': [infos]
            'onetime_info': one_time 
            'infos': [info_1, ...],
            'n_step': n
        }

        Requires:
            @self.learner_config.algo.n_step: n, number of steps per experience
            @self.learner_config.algo.stride: after sending experience [state_i, ...]
            the next experience is [state_{i + stride}]
    """
    def __init__(self, env, learner_config, session_config):
        super().__init__(env, learner_config, session_config)
        self._ob = None  # obs of the current time step
        self.n_step = self.learner_config.algo.n_step
        self.stride = self.learner_config.algo.stride # Stride for moving window
        if self.stride < 1:
            raise ConfigError('stride {} for experience generation cannot be less than 1'.format(self.learner_config.algo.stride))
        self.last_n = deque()

    def _reset(self):
        self._ob, info = self.env.reset()
        self.last_n.clear()
        return self._ob, info

    def _step(self, action):
        action_choice, action_info = action
        obs_next, reward, done, info = self.env.step(action_choice)
        self.last_n.append([self._ob, action_choice, reward, done, action_info[0], action_info[1], info])

        if len(self.last_n) == self.n_step:
            self.send(self.last_n, obs_next)
            for i in range(self.stride):
                if len(self.last_n) > 0:
                    self.last_n.popleft()
        self._ob = obs_next
        return obs_next, reward, done, info

    def send(self, data, obs_next):
        obs, actions, rewards, dones, persistent_infos, infos = [], [], [], [], [], []
        hash_dict = {}
        nonhash_dict = {}
        onetime_infos = None
        for index, (ob, action, reward, done, onetime_info, persistent_info, info) in enumerate(data):
            # Store observations in a deduplicated way
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            persistent_infos.append(persistent_info)
            if onetime_infos == None: onetime_infos = onetime_info

        hash_dict = {
            'obs': obs,
            'obs_next': obs_next,
        }
        nonhash_dict = {
            'actions': actions,
            'onetime_infos' : onetime_infos,
            'persistent_infos': persistent_infos,
            'rewards': rewards,
            'dones': dones,
            'infos': infos,
            'n_step': len(data),
        }
        self.sender.send(hash_dict, nonhash_dict)


class ExpSenderWrapperMultiStepMovingWindow(ExpSenderWrapperMultiStep):
    """
        Base class for all classes that send experience in format
        {   
            'obs_arr': [state_1, ..., state_n]
            'obs_next': [state_{n + 1}]
            'action_arr': [action_1, ...],
            'reward_arr': [reward_1, ...],
            'done_arr': [done_1, ...],
            'info_arr': [info_1, ...],
            'n_step': n
        }

        Requires:
            @self.learner_config.algo.n_step: n, number of steps per experience
            @self.learner_config.algo.stride: after sending experience [state_i, ...]
            the next experience is [state_{i + stride}]
    """
    def __init__(self, env, learner_config, session_config):
        super().__init__(env, learner_config, session_config)
        self._obs = None  # obs of the current time step
        self.n_step = self.learner_config.algo.n_step
        self.stride = self.learner_config.algo.stride # Stride for moving window
        if self.stride < 1:
            raise ConfigError('stride {} for experience generation cannot be less than 1'.format(self.learner_config.algo.stride))
        self.last_n = deque()

    def _reset(self):
        self._obs, info = self.env.reset()
        self.last_n.clear()
        return self._obs, info

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self.last_n.append([self._obs, action, reward, done, info])
        if len(self.last_n) == self.n_step:
            self.send(self.last_n, obs_next)
            for i in range(self.stride):
                if len(self.last_n) > 0:
                    self.last_n.popleft()
        self._obs = obs_next
        return obs_next, reward, done, info


class ExpSenderWrapperMultiStepEpisode(ExpSenderWrapperMultiStep):
    """
        Base class for all classes that send experience in format
        {   
            'obs_arr': [state_1, ..., state_T]
            'obs_next': [None]
            'action_arr': [action_1, ...],
            'reward_arr': [reward_1, ...],
            'done_arr': [done_1, ...],
            'info_arr': [info_1, ...],
            'n_step': T
        }

        T is episode length
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
            self.send(self.trajectory, None)
            self.trajectory.clear()
        self._obs = obs_next
        return obs_next, reward, done, info
