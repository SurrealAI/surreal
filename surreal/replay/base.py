import threading
import time
import surreal.utils as U
from surreal.session import get_tensorplex_client, get_loggerplex_client
from surreal.distributed import ExpQueue, ZmqServer


replay_registry = {}

def register_replay(target_class):
    replay_registry[target_class.__name__] = target_class

def replay_factory(replay_name):
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

        self._exp_queue = ExpQueue(
            host=self.session_config.replay.collector_backend_host,
            port=self.session_config.replay.collector_backend_port,
            max_size=self.session_config.replay.max_puller_queue,
            exp_handler=self._insert_wrapper,
        )
        self._sampler_server = ZmqServer(
            host=self.session_config.replay.sampler_backend_host,
            port=self.session_config.replay.sampler_backend_port,
            handler=self._sample_request_handler,
            is_pyobj=True,
            load_balanced=True,
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
        self.log = get_loggerplex_client(
            '{}/{}'.format('replay', self.index),
            self.session_config
        )
        self.tensorplex = get_tensorplex_client(
            '{}/{}'.format('replay', self.index),
            self.session_config
        )
        self.tensorplex_core = get_tensorplex_client(
            '{}/{}'.format('replay_core', self.index),
            self.session_config
        )
        self.tensorplex_system = get_tensorplex_client(
            '{}/{}'.format('replay_system', self.index),
            self.session_config
        )
        self._tensorplex_thread = None
        self._has_tensorplex = self.session_config.replay.tensorboard_display

        # Number of experience collected by agents
        self.cumulative_collected_count = 0
        # Number of experience sampled by learner
        self.cumulative_sampled_count = 0
        # Number of sampling requests from the learner
        self.cumulative_request_count = 0
        # Timer for tensorplex reporting
        self.last_tensorplex_iter_time = None
        
        self.insert_time = U.TimeRecorder(decay=0.99998)
        self.sample_time = U.TimeRecorder()

        # moving avrage of about 100s
        self.exp_in_speed = U.MovingAverageRecorder(decay=0.99)
        self.exp_out_speed = U.MovingAverageRecorder(decay=0.99)
        self.handle_sample_request_speed = U.MovingAverageRecorder(decay=0.99)

    def _insert_wrapper(self, exp_dict):
        """
            Allows us to do some book keeping in the base class
        """
        self.cumulative_collected_count += 1
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
        self.cumulative_request_count += 1
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
        self.init_time = init_time
        self.last_experience_count = 0
        self.last_sample_count = 0
        self.last_request_count = 0
        self.last_tensorplex_iter_time = time.time()
        while True:
            time.sleep(1.)
            self.generate_tensorplex_report()
            self.last_tensorplex_iter_time = time.time()

    def generate_tensorplex_report(self):
        """ 
            Generates tensorplex reports
        """
        global_step=int(time.time() - self.init_time)

        time_elapsed = time.time() - self.last_tensorplex_iter_time + 1e-6

        cum_count_collected = self.cumulative_collected_count
        new_exp_count = cum_count_collected - self.last_experience_count
        self.last_experience_count = cum_count_collected

        cum_count_sampled = self.cumulative_sampled_count
        new_sample_count = cum_count_sampled - self.last_sample_count
        self.last_sample_count = cum_count_sampled

        cum_count_requests = self.cumulative_request_count
        new_request_count = cum_count_requests - self.last_request_count
        self.last_request_count = cum_count_requests

        exp_in_speed = self.exp_in_speed.add_value(new_exp_count / time_elapsed)
        exp_out_speed = self.exp_out_speed.add_value(new_sample_count / time_elapsed)
        handle_sample_request_speed = self.handle_sample_request_speed.add_value(
                                                    new_request_count / time_elapsed)

        insert_time = self.insert_time.avg
        sample_time = self.sample_time.avg
        serialize_time = self._sampler_server.serialize_time.avg

        core_metrics = {
            'num_exps': len(self),
            'total_collected_exps': cum_count_collected,
            'total_sampled_exps': cum_count_sampled,
            'total_sample_requests': self.cumulative_request_count,
            'exp_in_per_s': exp_in_speed,
            'exp_out_per_s': exp_out_speed,
            'requests_per_s': handle_sample_request_speed,
            'insert_time_s': insert_time,
            'sample_time_s': sample_time,
            'serialize_time_s': serialize_time,
        }

        serialization_load = serialize_time * handle_sample_request_speed / time_elapsed
        collect_exp_load = insert_time * exp_in_speed / time_elapsed
        sample_exp_load = sample_time * handle_sample_request_speed / time_elapsed
        
        system_metrics = {
            'lifetime_experience_utilization_percent': \
                cum_count_sampled / (cum_count_collected + 1) * 100,
            'current_experience_utilization_percent': exp_out_speed / (exp_in_speed + 1) * 100,
            'serialization_load_percent': serialization_load * 100,
            'collect_exp_load_percent': collect_exp_load * 100,
            'sample_exp_load_percent': sample_exp_load * 100,
            'exp_queue_occupancy_percent': self._exp_queue.occupancy() * 100,
        }

        self.tensorplex_core.add_scalars(core_metrics, global_step=global_step)
        self.tensorplex_system.add_scalars(system_metrics, global_step=global_step)

