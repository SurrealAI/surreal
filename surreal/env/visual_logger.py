import time
import random
import gym
from pycrayon import CrayonClient
from .wrapper import Wrapper

class TensorboardWrapper(Wrapper):
    '''
    class for better visualizing training curve. 
    This visualizer logs:
        cumulative reward every episode
        number of iterations every second (to check for bottlenecks)
        number of steps per episod
    '''
    HOST_IP = None
    CLIENT  = None

    @classmethod
    def build(cls, hostIP, num_plots_per_group):
        '''
        class method to be called once to initialize 
        Tensorbord experiments
        '''
        client = CrayonClient(hostname=hostIP)

        # first clear history
        client.remove_all_experiments()

        # Hack to get around pycrayon bug where it cannot find empty experiments
        no_fault = True
        exp_no = 1
        while no_fault:
            try:
                client.remove_experiment('experiment_{}'.format(exp_no))
                exp_no += 1
                print(exp_no)
            except:
                no_fault = False

        for i in range(num_plots_per_group):
            client.create_experiment('experiment_{}'.format(i + 1))

        cls.HOST_IP = hostIP
        cls.CLIENT  = client

    def __init__(self, env, groupID, plotID):
        super().__init__(env)
        self.env = env
        self.eps_reward = 0
        self.eps_length = 0
        self.itr_per_sec = 0
        self.start_time = time.time()
        self.step_exec  = 0

        self.client = CrayonClient(hostname=TensorboardWrapper.HOST_IP)
        self.experiment = self.client.open_experiment("experiment_{}".format(plotID + 1))
        self.groupID = groupID

    def _reset(self, **kwargs):
        self.eps_reward = 0
        self.eps_length = 0
        self.start_time = time.time()
        self.step_exec  = 0
        return self.env.reset(**kwargs)

    def _step(self, action):
        state, step_reward, terminal, info = self.env.step(action)

        self.eps_reward += step_reward
        self.eps_length += 1
        self.step_exec  += 1

        # logging iterations per second 
        elapsed = time.time() - self.start_time
        if elapsed >= 1:
            self.experiment.add_scalar_value("group_{}_iter_per_sec".format(self.groupID), 
                                             self.step_exec / elapsed)
            self.start_time = time.time()
            self.step_exec  = 0

        if terminal:
            # logging per episode statistics (reward, episode length, training speed)
            self.experiment.add_scalar_value("group_{}_reward".format(self.groupID), self.eps_reward)
            self.experiment.add_scalar_value("group_{}_eps_length".format(self.groupID), self.eps_length)

        return state, step_reward, terminal, info

    def log(name, value):
        '''
        Just in case we want to log info manually
        '''
        self.experiment.add_scalar_value("group_{}_{}".format(self.groupID, name), value)

    def save_run(self, fname):
        '''
        Saving tensorflow log to zipfile
        '''
        self.experiment.to_zip(filename=fname)

    def load_run(self, fnames):
        '''
        reading from zipfile into tensorboard
        '''
        for exp_name, path in fnames:
            try: self.client.remove_experiment(exp_name)
            except: pass
            self.client.create_experiment(exp_name, zip_file=path)
