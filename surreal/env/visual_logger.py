import time
import random
import gym
from pycrayon import CrayonClient
from .wrapper import Wrapper


class TensorBoardMonitor(object):
    def __init__(self, hostIP, port = 8889):
        self.hostIP = hostIP
        self.port   = port
        self.client = CrayonClient(hostname=self.hostIP, port=self.port)
        self.exp = None

    @classmethod
    def build_experiments(cls, hostIP, port=8889, num_plots_per_group = 1):
        '''
        method to be called once to initialize Tensorbord experiments.
        Restrict usage to agent side before forking out agent threads/process

        CAUTION: calling this again would erase all previous data logged
        this is due to that crayon API does not remove empty experiments 
        properly. Make sure this only gets called ONCE per experiment.
        Also make sure this is executed before agents are created
        '''

        # first create crayon client
        client = CrayonClient(hostname=hostIP, port=port)

        # first clear history
        client.remove_all_experiments()

        # Hack to get around pycrayon bug where it cannot find empty experiments
        for i in range(num_plots_per_group):
            try: 
                client.create_experiment('agents_run_{}'.format(i + 1))
            except:
                client.remove_experiment('agents_run_{}'.format(i + 1))
                client.create_experiment('agents_run_{}'.format(i + 1))

        try: 
            client.create_experiment('master')
        except: 
            client.remove_experiment('master')
            client.create_experiment('master')
            
    def add_scalar_values(self, names, values, wall_time=-1, step=-1):
        if self.exp is None:
            self.exp = self.client.open_experiment('master')

        if isinstance(names, list):
            assert(len(names) == len(values)), "Logging names and values must have same length"
            val_dict = {}
            for name, val in zip(names, values):
                val_dict[name] = val
            self.exp.add_scalar_dict(val_dict, wall_time=wall_time, step=step)
            
        else: 
            # case of input is single scalar value
            self.exp.add_scalar_value(names, values, wall_time=wall_time, step=step)

    def add_histogram_value(self, name, hist, tobuild=False, wall_time=-1, step=-1):
        if self.exp is None:
            self.exp = self.client.open_experiment('master')
        self.exp.add_histogram_value(name, hist, tobuild=tobuild, wall_time=wall_time, step=step)

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


class TensorboardWrapper(Wrapper):
    '''
    class for better visualizing training curve. 
    This visualizer logs:
        cumulative reward every episode
        number of iterations every second (to check for bottlenecks)
        number of steps per episod
    '''
    def __init__(self, env, host, groupID, plotID):
        super().__init__(env)
        self.env = env

        self.eps_reward = 0
        self.eps_length = 0
        self.itr_per_sec = 0
        self.start_time = time.time()
        self.step_exec  = 0

        client = CrayonClient(hostname=host)
        self.experiment = client.open_experiment("agents_run_{}".format(plotID + 1))
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
            # logging per episode statistics (reward, episode length)
            self.experiment.add_scalar_dict( {"group_{}_reward".format(self.groupID): self.eps_reward,
                                              "group_{}_eps_length".format(self.groupID): self.eps_length})

        return state, step_reward, terminal, info

    def log(name, value, wall_time=-1, step=-1):
        '''
        Just in case we want to log info manually
        '''
        self.experiment.add_scalar_value("group_{}_{}".format(self.groupID, name), value, wall_time=-1, step=-1)
