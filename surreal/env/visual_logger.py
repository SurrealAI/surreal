import time
import random
import gym
from pycrayon import CrayonClient
from .wrapper import Wrapper


class TensorboardMonitor(object):
    '''
    This class implements support for multi-node Tensorboard using Crayon library
    It contains the following function:
        constructor
        add_scalar_values
        add_histogram_value
        save_run
        load_run

        build_experiments

    '''
    def __init__(self, hostIP, port = 8889):
        '''
        Constructor for TensorboardMonitor class
        Args:
            hostIP (str): IP address of the tensorboard host server
            port (int, optional): port to which to send data

        properties of the object:
            hostIP
            port 
            client: CrayonClient object that maintains connection with tensorboard server
            exp: experiment to which this monitor logs to. 
                 Default to "master" on learner side
        '''
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

        Args:
            hostIP (str): IP address of the tensorboard host server
            port (int, optional): port to which to send data
            num_plots_per_group (int): how many plot per group, corresponds to
                                       number of experiments
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
        '''
        Function to add one or many values to experiment
        Args:
            name (str/list<str>): name(s) of the scalar variable to add value to
            values (float/list<float>): value(s) to be added 
            wall_time(float, optional): time elapsed for tensorboard display 
            step (int, optional): number of step in the experiment for tensorboard display

        Notes on step and wall_time. In tensorboard, data can either displayed by step,
            meaning that each data point is logged after another and displayed with 
            equal spacing regardless of how much time have passed in between. Or data can be
            displayed by wall_time, which is the actual time passed among logs.

        If not specified, the wall_time will be set to the current time and the step to the 
        step of the previous point with this name plus one (or 0 if its the first point with this name)
        '''

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
        '''
        Function to add histogram data to experiment
        Args:
            name (str/list<str>): name of the histogram variable to add value to
            hist (dict<str:float>/list<float>): histogram to be added 
            tobuild (bool, optoinal): option to build
            wall_time(float, optional): time elapsed for tensorboard display 
            step (int, optional): number of step in the experiment for tensorboard display
        
        Note on tobuild:
        If tobuild is False, hist should be a dictionary containing: 
        {
            "min": minimum value, 
            "max": maximum value, 
            "num": number of items in the histogram, 
            "bucket_limit": a list with the right limit of each bucket, 
            "bucker": a list with the number of element in each bucket, 
            "sum": optional, the sum of items in the histogram, 
            "sum_squares": optional, the sum of squares of items in the histogram
        }
        If tobuild if True, hist should be a list of value from which 
        an histogram is going to be built.

        If not specified, the wall_time will be set to the current time and the step to the 
        step of the previous point with this name plus one (or 0 if its the first point with this name)
        
        '''
        if self.exp is None:
            self.exp = self.client.open_experiment('master')
        self.exp.add_histogram_value(name, hist, tobuild=tobuild, wall_time=wall_time, step=step)

    def save_run(self, fname):
        '''
        Saving tensorflow log to zipfile
        Args:
            fname (str): path of the data to be saved to
        '''
        self.experiment.to_zip(filename=fname)

    def load_run(self, fnames):
        '''
        reading from zipfile into tensorboard
        Args: 
            fnames (list<(str, str)>): list of tuples of strings,
                    first is experiment name and second is path for
                    experiment zip file
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
