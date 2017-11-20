import time
import random
import gym
from wrapper import Wrapper
from tensorplex.distributed_writer import make_writer_client

class TensorboardWrapper(Wrapper):
    '''
    class for better visualizing training curve. 
    This visualizer logs:
        cumulative reward every episode
        number of iterations every second (to check for bottlenecks)
        number of steps per episod
    '''
    def __init__(self, env, setID, groupID, host='localhost'):
        super().__init__(env)
        self.env = env

        self.eps_reward = 0
        self.eps_length = 0
        self.itr_per_sec = 0
        self.start_time = time.time()
        self.step_exec  = 0

        self.setID = setID
        self.groupID = groupID

        self.global_step_counter = 0
        self.global_eps_counter = 0

        self.writer = make_writer_client(False, None, host, port=6379, queue_name='remotecall')

    def _reset(self, **kwargs):
        self.eps_reward = 0
        self.eps_length = 0
        self.start_time = time.time()
        self.step_exec  = 0

        return self.env.reset(**kwargs)


    def _step(self, action):

        state, step_reward, terminal, info = self.env.step(action)
        self.global_step_counter += 1

        self.eps_reward += step_reward
        self.eps_length += 1
        self.step_exec  += 1

        # logging iterations per second 
        elapsed = time.time() - self.start_time
        if elapsed >= 1:
            self.writer.add_scalar(self.setID, "group_{}/iter_per_sec".format(self.groupID), 
                                            self.step_exec / elapsed, global_step=self.global_step_counter)
            self.start_time = time.time()
            self.step_exec  = 0

        if terminal:
            # logging per episode statistics (reward, episode length)
            self.global_eps_counter += 1
            self.writer.add_scalar(self.setID, "group_{}/reward".format(self.groupID), 
                                   self.eps_reward, global_step=self.global_eps_counter)
            self.writer.add_scalar(self.setID, "group_{}/eps_length".format(self.groupID), 
                                   self.eps_length, global_step=self.global_eps_counter)

        return state, step_reward, terminal, info

