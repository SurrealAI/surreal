import threading
import time
import surreal.utils as U
from surreal.session import (Loggerplex, TensorplexClient, Config, extend_config,
                             BASE_SESSION_CONFIG, BASE_ENV_CONFIG)
from surreal.distributed import ExpQueue, ZmqServer


replay_registry = {}

def register_replay(target_class):
    replay_registry[target_class.__name__] = target_class

def replayFactory(replay_name):
    return replay_registry[replay_name]

class ReplayMeta(type):
    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        register_replay(cls)
        return cls

class Replay(object, metaclass=ReplayMeta):
    """
        Important: When extending this class, make sure to follow the init method signature so that 
        orchestrating functions can properly initialize the replay server.
    """
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 index=0):
        """
        """
        # Note that there're 2 replay configs:
        # one in learner_config that controls algorithmic part of the replay logic
        # one in session_config that controls system settings
        self.learner_config = learner_config
        self.env_config = env_config
        self.session_config = session_config
        self.index = index

        exp_queue_add = "tcp://{}:{}".format(self.session_config.replay.collector_backend_host,
                                            self.session_config.replay.collector_backend_port)
        self._exp_queue = ExpQueue(
            address=exp_queue_add,
            max_size=self.session_config.replay.max_puller_queue,
            exp_handler=self._insert_wrapper,
        )
        self._sampler_server = ZmqServer(
            host=self.session_config.replay.sampler_backend_host,
            port=self.session_config.replay.sampler_backend_port,
            handler=self._sample_request_handler,
            is_pyobj=True,
            loadbalanced=True,
        )
        self._job_queue = U.JobQueue()

        self._evict_interval = self.session_config.replay.evict_interval
        self._evict_thread = None

        self._setup_logging()

    def start_threads(self):
        if self._has_tensorplex:
            self.start_tensorplex_thread()
        
        self._exp_queue.start_dequeue_thread()
        self._exp_queue.start_enqueue_thread()
        
        if self._evict_interval:
            self.start_evict_thread()

        self._sampler_server.start()
        self._sampler_server.join()

    def insert(self, exp_dict):
        """
        Add a new experience to the replay.
        Includes passive evict logic if memory capacity is exceeded.

        Args:
            exp_dict: {[obs], action, reward, done, info}
        """
        raise NotImplementedError

    def sample(self, batch_size):
        """
        This function is called in _sample_handler for learner side Zmq request

        Args:
            batch_size:

        Returns:
            a list of exp_tuples
        """
        raise NotImplementedError

    def evict(self):
        """
        Actively evict old experiences.
        """
        pass

    def start_sample_condition(self):
        """
        Tells the thread to start sampling only when this condition is met.
        For example, only when the replay memory has > 10K experiences.

        Returns:
            bool: whether to start sampling or not
        """
        raise NotImplementedError


    def __len__(self):
        raise NotImplementedError

    # ======================== internal methods ========================    
    def _setup_logging(self):
        self.log = Loggerplex(
            name='{}/{}'.format('replay', self.index),
            session_config=self.session_config
        )
        self.tensorplex = TensorplexClient(
            '{}/{}'.format('replay', self.index),
            host=self.session_config.tensorplex.host,
            port=self.session_config.tensorplex.port,
        )
        self._tensorplex_thread = None
        self._has_tensorplex = self.session_config.replay.tensorboard_display

        # Number of experience collected by agents
        self.cumulative_experience_count = 0
        # Number of experience sampled by learner
        self.cumulative_sampled_count = 0
        
        self.insert_time = U.TimeRecorder()
        self.sample_time = U.TimeRecorder()

    def _insert_wrapper(self, exp_dict):
        """
            Allows us to do some book keeping in the base class
        """
        self.cumulative_experience_count += 1
        with self.insert_time.time():
            self.insert(exp_dict)

    def _sample_request_handler(self, batch_size):
        """
        Handle requests to the learner
        https://stackoverflow.com/questions/29082268/python-time-sleep-vs-event-wait
        Since we don't have external notify, we'd better just use sleep
        """
        U.assert_type(batch_size, int)
        while not self.start_sample_condition():
            time.sleep(0.01)
        self.cumulative_sampled_count += batch_size
        with self.sample_time.time():
            return self.sample(batch_size)

    def start_evict_thread(self):
        if self._evict_thread is not None:
            raise RuntimeError('evict thread already running')
        self._evict_thread = U.start_thread(self._evict_loop)
        return self._evict_thread

    def _evict_loop(self):
        assert self._evict_interval
        while True:
            time.sleep(self._evict_interval)
            self.evict()

    def start_tensorplex_thread(self):
        if self._tensorplex_thread is not None:
            raise RuntimeError('tensorplex thread already running')
        self._tensorplex_thread = U.start_thread(
            self._tensorplex_loop,
            args=(time.time(),)
        )
        return self._tensorplex_thread

    def _tensorplex_loop(self, init_time):
        self.last_experience_count = 0
        self.last_sample_count = 0
        while True:
            time.sleep(1.)
            self.tensorplex.add_scalars(self.get_tensorplex_report_dict(),
                global_step=int(time.time() - init_time))

    def get_tensorplex_report_dict(self):
        """ 
            Returns a dictionary containing data to be reported to tensorplex
        """
                    
        cum_count_exp = self.cumulative_experience_count
        new_exp_count = cum_count_exp - self.last_experience_count
        self.last_experience_count = cum_count_exp

        cum_count_sample = self.cumulative_sampled_count
        new_sample_count = cum_count_sample - self.last_sample_count
        self.last_sample_count = cum_count_sample

        return {
            'exp_queue_occupancy': self._exp_queue.occupancy(),
            'num_exps': len(self),
            'cumulative_exps': self.cumulative_experience_count,
            'exp_in/s': new_exp_count,
            'exp_out/s': new_sample_count,
            'insert_time': self.insert_time.avg,
            'sample_time': self.sample_time.avg,
            'serialize_time': self._sampler_server.serialize_time.avg,
        }
