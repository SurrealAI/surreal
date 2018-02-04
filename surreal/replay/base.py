import threading
import time
import surreal.utils as U
from surreal.session import (Loggerplex, StatsTensorplex, Config, extend_config,
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

class ReplayCore(object, metaclass=ReplayMeta):
    def __init__(self, *,
                 puller_port,
                 sampler_port,
                 max_puller_queue,
                 evict_interval):
        """
        Args:
            puller_port: server, pull from agent side
            sampler_port: server, send to learner side
            max_puller_queue
            evict_interval: in seconds, 0 to disable evict
        """
        self._exp_queue = ExpQueue(
            port=puller_port,
            max_size=max_puller_queue,
            exp_handler=self._insert_base,
        )
        self._sampler_server = ZmqServer(
            port=sampler_port,
            handler=self._sample_request_handler,
            is_pyobj=True
        )
        self._evict_interval = evict_interval
        self._evict_thread = None
        self.cumulative_experience_count = 0
        self.cumulative_sampled_count = 0
        # self._sample_condition = threading.Condition()

    def start_threads(self):
        self._exp_queue.start_dequeue_thread()
        self._exp_queue.start_enqueue_thread()
        if self._evict_interval:
            self.start_evict_thread()
        self._sampler_server.run_loop(block=True)



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
    def _insert_base(self, exp_dict):
        """
            Allows us to do some book keeping in the base class
        """
        self.cumulative_experience_count += 1
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
        return self.sample(batch_size)

    def _evict_loop(self):
        assert self._evict_interval
        while True:
            time.sleep(self._evict_interval)
            self.evict()

    def start_evict_thread(self):
        if self._evict_thread is not None:
            raise RuntimeError('evict thread already running')
        self._evict_thread = U.start_thread(self._evict_loop)
        return self._evict_thread


class Replay(ReplayCore):
    """
        Important: When extending this class, make sure to follow the init method signature so that 
        orchestrating functions can properly initialize the replay server.
    """
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config):
        """
        """
        # Note that there're 2 replay configs:
        # one in learner_config that controls algorithmic part of the replay logic
        # one in session_config that controls system settings
        self.learner_config = Config(learner_config)
        self.replay_config = self.learner_config.replay
        self.replay_config.extend(self.default_config())
        self.env_config = extend_config(env_config, BASE_ENV_CONFIG)
        self.session_config = extend_config(session_config, BASE_SESSION_CONFIG)
        self.replay_config.update(self.session_config.replay)

        super().__init__(
            puller_port=self.replay_config.port,
            sampler_port=self.replay_config.sampler_port,
            max_puller_queue=self.replay_config.max_puller_queue,
            evict_interval=self.replay_config.evict_interval
        )
        self.log = Loggerplex(
            name='replay',
            session_config=self.session_config
        )
        self.tensorplex = StatsTensorplex(
            section_name='replay',
            session_config=self.session_config
        )
        self._tensorplex_thread = None
        self._has_tensorplex = self.replay_config.tensorboard_display
        self._job_queue = U.JobQueue()

    def start_threads(self):
        if self._has_tensorplex:
            self.start_tensorplex_thread()
        super().start_threads()

    def default_config(self):
        """
        Returns:
            dict of default configs, will be placed in learner_config['replay']
        """
        return {
            'batch_size': '_int_',
        }

    def _insert_base(self, exp_dict):
        self.insert(exp_dict)

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
            
            cum_count_exp = self.cumulative_experience_count
            new_exp_count = cum_count_exp - self.last_experience_count
            self.last_experience_count = cum_count_exp

            cum_count_sample = self.cumulative_sampled_count
            new_sample_count = cum_count_sample - self.last_sample_count
            self.last_sample_count = cum_count_sample

            self.tensorplex.add_scalars({
                'exp_queue_occupancy': self._exp_queue.occupancy(),
                'num_exps': len(self),
                'cumulative_exps': self.cumulative_experience_count,
                'exp_in/s': new_exp_count,
                'exp_out/s': new_sample_count,
            }, global_step=int(time.time() - init_time))
