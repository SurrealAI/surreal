import threading
import time
import surreal.utils as U
from surreal.session import (Loggerplex, StatsTensorplex, Config, extend_config,
                             BASE_SESSION_CONFIG, BASE_ENV_CONFIG)
from .aggregator import torch_aggregate
from surreal.distributed import ExpQueue, ZmqServer


class ReplayCore(object):
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
            exp_handler=self.insert,
        )
        self._sampler_server = ZmqServer(
            port=sampler_port,
            handler=self._sample_request_handler,
            is_pyobj=True
        )
        self._evict_interval = evict_interval
        self._evict_thread = None
        # self._sample_condition = threading.Condition()

    def start_threads(self):
        self._exp_queue.start_dequeue_thread()
        self._exp_queue.start_enqueue_thread()
        if self._evict_interval:
            self.start_evict_thread()
        self._sampler_server.run_loop(block=True)

    def insert(self, exp_tuple):
        """
        Add a new experience to the replay.
        Includes passive evict logic if memory capacity is exceeded.

        Args:
            exp_tuple: ExpTuple([obs], action, reward, done, info)
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

    def aggregate_batch(self, exp_list):
        """
        Will be called in `next_batch()` method to produce the actual inputs
        to the neural network training loop.

        Args:
            exp_list: list of ExpTuple (from `_sample()`)
                [obs, reward, action, done, info]

        Returns:
            batched Tensors, batched action/reward vectors, etc.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    # ======================== internal methods ========================
    def _sample_request_handler(self, batch_size):
        """
        Handle requests to the learner
        https://stackoverflow.com/questions/29082268/python-time-sleep-vs-event-wait
        Since we don't have external notify, we'd better just use sleep
        """
        U.assert_type(batch_size, int)
        while not self.start_sample_condition():
            time.sleep(0.01)
        return self.aggregate_batch(self.sample(batch_size))

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
    def __init__(self,
                 learn_config,
                 env_config,
                 session_config):
        """
        """
        # Note that there're 2 replay configs:
        # one in learner_config that controls algorithmic part of the replay logic
        # one in session_config that controls system settings
        self.replay_config = Config(learn_config).replay
        self.replay_config.extend(self.default_config())
        self.env_config = extend_config(env_config, BASE_ENV_CONFIG)
        self.session_config = extend_config(session_config, BASE_SESSION_CONFIG)

        super().__init__(
            puller_port=self.session_config.replay.port,
            sampler_port=self.session_config.replay.sampler_port,
            max_puller_queue=self.session_config.replay.max_puller_queue,
            evict_interval=self.session_config.replay.evict_interval
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
        self._has_tensorplex = self.session_config.replay.tensorboard_display
        self._job_queue = U.JobQueue()

    def start_threads(self):
        if self._has_tensorplex:
            self.start_tensorplex_thread()
        super().start_threads()

    def default_config(self):
        """
        Returns:
            dict of default configs, will be placed in learn_config['replay']
        """
        return {
            'batch_size': '_int_',
        }

    def aggregate_batch(self, exp_list):
        return torch_aggregate(
            exp_list,
            obs_spec=self.env_config.obs_spec,
            action_spec=self.env_config.action_spec,
        )

    def start_tensorplex_thread(self):
        if self._tensorplex_thread is not None:
            raise RuntimeError('tensorplex thread already running')
        self._tensorplex_thread = U.start_thread(
            self._tensorplex_loop,
            args=(time.time(),)
        )
        return self._tensorplex_thread

    def _tensorplex_loop(self, init_time):
        while True:
            time.sleep(1.)
            self.tensorplex.add_scalars({
                'queue_occupancy': self._exp_queue.occupancy(),
                'num_of_experiences': len(self),
            }, global_step=int(time.time() - init_time))
