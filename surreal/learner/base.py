"""
Template class for all learners
"""
import surreal.utils as U
from surreal.session import (
    extend_config, PeriodicTracker, PeriodicTensorplex,
    BASE_ENV_CONFIG, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG
)
from surreal.session import StatsTensorplex, Loggerplex
from surreal.distributed import ZmqClient, ParameterPublisher
import queue
from easydict import EasyDict
import time

class PrefetchBatchQueue(object):
    """
    Pre-fetch a batch of exp from sampler on Replay side
    """
    def __init__(self,
                 sampler_host,
                 sampler_port,
                 batch_size,
                 max_size,):
        self._queue = queue.Queue(maxsize=max_size)
        self._batch_size = batch_size
        self._client = ZmqClient(
            host=sampler_host,
            port=sampler_port,
            is_pyobj=True
        )
        self._enqueue_thread = None

        self.cum_requests = 0
        self.cum_time = 0
        self.ctr = 0

    def _enqueue_loop(self):
        while True:
            pre_time = time.time()
            sample = self._client.request(self._batch_size)
            post_time = time.time()

            self.ctr += 1
            self.cum_requests *= 0.99
            self.cum_requests += 1
            self.cum_time *= 0.99
            self.cum_time += post_time - pre_time
            self.ctr += 1
            if self.ctr % 100 == 0:
                print('[Fetch Experience] {:.2f} ms'.format(self.cum_time / self.cum_requests * 1000))
            self._queue.put(sample, block=True)

    def start_enqueue_thread(self):
        """
        Producer thread, runs sampler function on a priority replay structure
        Args:
            sampler: function batch_i -> list
                returns exp_dicts with 'obs_pointers' field
            start_sample_condition: function () -> bool
                begins sampling only when this returns True.
                Example: when the replay memory exceeds a threshold size
            start_sample_condvar: threading.Condition()
                notified by Replay.insert() when start sampling condition is met
            evict_lock: do not evict in the middle of fetching exp, otherwise
                we might fetch a null exp that just got evicted.
                locked by Replay.evict()
        """
        if self._enqueue_thread is not None:
            raise RuntimeError('Enqueue thread is already running')
        self._enqueue_thread = U.start_thread(self._enqueue_loop)
        return self._enqueue_thread

    def dequeue(self):
        """
        Called by the neural network, draw the next batch of experiences
        """
        return self._queue.get(block=True)

    def __len__(self):
        return self._queue.qsize()


learner_registry = {}

def register_learner(target_class):
    learner_registry[target_class.__name__] = target_class

def learnerFactory(learner_name):
    return learner_registry[learner_name]

class LearnerMeta(U.AutoInitializeMeta):
    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        register_learner(cls)
        return cls

class LearnerCore(metaclass=LearnerMeta):
    def __init__(self, *,
                 sampler_host,
                 sampler_port,
                 ps_publish_port,
                 batch_size,
                 max_prefetch_batch_queue):
        """
        Write log to self.log

        Args:
            sampler_host: client to connect to replay node sampler
            sampler_port: client to connect to replay node
            ps_pub_port: parameter server PUBLISH port
        """
        self._ps_publisher = None  # in _initialize()
        self._ps_port = ps_publish_port
        self._prefetch_queue = PrefetchBatchQueue(
            sampler_host=sampler_host,
            sampler_port=sampler_port,
            batch_size=batch_size,
            max_size=max_prefetch_batch_queue,
        )

    def _initialize(self):
        """
        For AutoInitializeMeta interface
        """
        self._ps_publisher = ParameterPublisher(
            port=self._ps_port,
            module_dict=self.module_dict()
        )
        self._prefetch_queue.start_enqueue_thread()

    def default_config(self):
        """
        Returns:
            a dict of defaults.
        """
        return BASE_LEARNER_CONFIG

    def learn(self, batch_exp):
        """
        Abstract method runs one step of learning

        Args:
            batch_exp: batched experience, format is a list of whatever experience sender wrapper returns

        Returns:
            td_error or other values for prioritized replay
        """
        raise NotImplementedError

    def module_dict(self):
        """
        Dict of modules to be broadcasted to the parameter server.
        MUST be consistent with the agent's `module_dict()`
        """
        raise NotImplementedError

    def save(self, file_path):
        """
        Checkpoint to disk
        """
        raise NotImplementedError

    def publish_parameter(self, iteration, message=''):
        """
        Learner publishes latest parameters to the parameter server.

        Args:
            iteration: the current number of learning iterations
            message: optional message, must be pickleable.
        """
        self._ps_publisher.publish(iteration, message=message)

    def fetch_batch(self):
        return self._prefetch_queue.dequeue()

    def fetch_iterator(self):
        while True:
            yield self.fetch_batch()


class Learner(LearnerCore):
    """
        Important: When extending this class, make sure to follow the init method signature so that 
        orchestrating functions can properly initialize the learner.
    """
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config):
        """
        Write log to self.log

        Args:
            config: a dictionary of hyperparameters. It can include a special
                section "log": {logger configs}
            model: utils.pytorch.Module for the policy network
        """
        self.learner_config = extend_config(learner_config, self.default_config())
        self.env_config = extend_config(env_config, BASE_ENV_CONFIG)
        self.session_config = extend_config(session_config, BASE_SESSION_CONFIG)
        super().__init__(
            sampler_host=self.session_config.replay.sampler_host,
            sampler_port=self.session_config.replay.sampler_port,
            ps_publish_port=self.session_config.ps.publish_port,
            batch_size=self.learner_config.replay.batch_size,
            max_prefetch_batch_queue=self.session_config.replay.max_prefetch_batch_queue
        )
        self.log = Loggerplex(
            name='learner',
            session_config=self.session_config
        )
        self.tensorplex = StatsTensorplex(
            section_name='learner',
            session_config=self.session_config
        )
        self._periodic_tensorplex = PeriodicTensorplex(
            tensorplex=self.tensorplex,
            period=self.session_config.tensorplex.update_schedule.learner,
            is_average=True,
            keep_full_history=False
        )

    def default_config(self):
        """
        Returns:
            a dict of defaults.
        """
        return BASE_LEARNER_CONFIG

    def update_tensorplex(self, tag_value_dict, global_step=None):
        """
        Args:
            tag_value_dict:
            global_step: None to use internal tracker value
        """
        self._periodic_tensorplex.update(tag_value_dict, global_step)
