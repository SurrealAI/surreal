"""
Template class for all learners
"""
import queue
import time
import surreal.utils as U
from surreal.session import (
    TimeThrottledTensorplex,
    get_loggerplex_client, get_tensorplex_client
)
from surreal.distributed import ZmqClient, ParameterPublisher, ZmqClientPool
from surreal.distributed.zmq_struct_new import LearnerDataPrefetcher

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
        self._client = ZmqClientPool(
            host=sampler_host,
            port=sampler_port,
            request=self._batch_size,
            handler=self._enqueue,
            is_pyobj=True,
        )
        self._enqueue_thread = None

        self.timer = U.TimeRecorder()

    def _enqueue(self, item):
        self._queue.put(item, block=True)            

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
        self._enqueue_thread = self._client
        self._client.start()
        return self._enqueue_thread

    def dequeue(self):
        """
        Called by the neural network, draw the next batch of experiences
        """
        with self.timer.time():
            return self._queue.get(block=True)

    def __len__(self):
        return self._queue.qsize()


learner_registry = {}


def register_learner(target_class):
    learner_registry[target_class.__name__] = target_class


def learner_factory(learner_name):
    return learner_registry[learner_name]


class LearnerMeta(U.AutoInitializeMeta):
    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        register_learner(cls)
        return cls


class Learner(metaclass=LearnerMeta):
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
        self.learner_config = learner_config
        self.env_config = env_config
        self.session_config = session_config

        self._setup_connection()
        self._setup_logging()
        self._setup_checkpoint()

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

    def checkpoint_attributes(self):
        """
        Returns:
            list of attributes to be tracked by checkpoint
        """
        return []

    ######
    # Internal, including Communication, etc.
    ######
    def _setup_connection(self):  
        # sampler_host = self.session_config.replay.sampler_frontend_host
        # sampler_port = self.session_config.replay.sampler_frontend_port
        ps_publish_port = self.session_config.ps.publish_port
        batch_size = self.learner_config.replay.batch_size
        # max_prefetch_batch_queue = self.session_config.learner.max_prefetch_batch_queue

        self._ps_publisher = None  # in _initialize()
        self._ps_port = ps_publish_port
        self._prefetch_queue = LearnerDataPrefetcher(
            session_config=self.session_config,
            batch_size=batch_size,
        )


    def _initialize(self):
        """
        For AutoInitializeMeta interface
        """
        # Module dict can only be acquired after subclass __init__
        self._ps_publisher = ParameterPublisher(
            port=self._ps_port,
            module_dict=self.module_dict()
        )
        self._prefetch_queue.start()
        self.start_tensorplex_thread()
        # restore_checkpoint should be called _after_ subclass __init__
        # that's why we put it in _initialize()
        if self.session_config.checkpoint.restore:
            self.restore_checkpoint()

    def publish_parameter(self, iteration, message=''):
        """
        Learner publishes latest parameters to the parameter server.

        Args:
            iteration: the current number of learning iterations
            message: optional message, must be pickleable.
        """
        self._ps_publisher.publish(iteration, message=message)

    def fetch_batch(self):
        return self._prefetch_queue.get()

    def fetch_iterator(self):
        while True:
            yield self.fetch_batch()


    ######
    # Logging
    ######
    def _setup_logging(self):
        self.learn_timer = U.TimeRecorder()
        # We don't do it here so that we don't require _prefetch_queue to be setup beforehands
        # self.fetch_timer = self._prefetch_queue.timer
        self.iter_timer = U.TimeRecorder()
        self.publish_timer = U.TimeRecorder()

        self.init_time = time.time()
        self.current_iter = 0

        self.last_time = self.init_time
        self.last_time_2 = self.init_time
        self.last_iter = 0

        self.log = get_loggerplex_client('learner', self.session_config)
        self.tensorplex = self._get_tensorplex('learner/learner')

        self._tensorplex_thread = None

    def _get_tensorplex(self, name):
        """
            Get the periodic tensorplex object
        Args:
            @name: The name of the collection of metrics
        """
        tp = get_tensorplex_client(
            name,
            self.session_config
        )
        update_schedule = self.session_config.tensorplex.update_schedule
        periodic_tp = TimeThrottledTensorplex(
            tensorplex=tp,
            min_update_interval=update_schedule.learner_min_update_interval,
        )
        return periodic_tp

    def start_tensorplex_thread(self):
        if self._tensorplex_thread is not None:
            raise RuntimeError('tensorplex thread already running')
        self._tensorplex_thread = U.PeriodicWakeUpWorker(target=self.generate_tensorplex_report)
        self._tensorplex_thread.start()
        return self._tensorplex_thread

    def generate_tensorplex_report(self):
        """
            Adds core and system level tensorplex stats
        """

        cur_time = time.time()
        current_iter = self.current_iter

        print('[tensorplex] iter: {}'.format(current_iter))
        print('[tensorplex] time: {}'.format(cur_time))

        iter_elapsed = current_iter - self.last_iter
        self.last_iter = current_iter

        global_step = cur_time - self.init_time
        time_elapsed = cur_time - self.last_time
        self.last_time = cur_time
        

        core_metrics = {}
        system_metrics = {}
        
        learn_time = self.learn_timer.avg + 1e-6
        fetch_timer = self._prefetch_queue.timer
        fetch_time = fetch_timer.avg + 1e-6
        iter_time = self.iter_timer.avg + 1e-6
        publish_time = self.publish_timer.avg + 1e-6
        # Time it takes to learn from a batch
        core_metrics['learn_time_s'] = learn_time
        # Time it takes to fetch a batch
        core_metrics['fetch_time_s'] = fetch_time
        # Time it takes to publish parameters
        core_metrics['publish_time_s'] = publish_time
        # Time it takes to complete one full iteration
        core_metrics['iter_time_s'] = iter_time

        # Number of iterations per second
        system_metrics['iter_per_s'] = iter_elapsed / time_elapsed
        # Percent of time spent on learning
        system_metrics['compute_load_percent'] = min(learn_time / iter_time * 100, 100)
        # Percent of time spent on IO
        system_metrics['io_fetch_experience_load_percent'] = min(fetch_time / iter_time * 100, 100)
        # Percent of time spent on publishing
        system_metrics['io_publish_load_percent'] = min(publish_time / iter_time * 100, 100)

        all_metrics = {}
        for k in core_metrics:
            all_metrics['.core/' + k] = core_metrics[k]
        for k in system_metrics:
            all_metrics['.system/' + k] = system_metrics[k]

        self.tensorplex.add_scalars(all_metrics)

    ######
    # Checkpoint
    ######
    def _setup_checkpoint(self):
        tracked_attrs = self.checkpoint_attributes()
        assert U.is_sequence(tracked_attrs), \
            'checkpoint_attributes must return a list of string attr names'
        self._periodic_checkpoint = U.PeriodicCheckpoint(
            U.f_join(self.session_config.folder, 'checkpoint'),
            name='learner',
            period=self.session_config.checkpoint.learner.periodic,
            tracked_obj=self,
            tracked_attrs=tracked_attrs,
            keep_history=self.session_config.checkpoint.learner.keep_history,
            keep_best=self.session_config.checkpoint.learner.keep_best  
            # TODO figure out how to add score to learner
        )

    def periodic_checkpoint(self, global_steps, score=None, **info):
        """
        Will only save at the end of each period

        Args:
            global_steps: 
            score: None when session_config.checkpoint.keep_best=False
            **info: other meta info you want to save in checkpoint metadata

        Returns:
            whether save() is actually called or not
        """
        return self._periodic_checkpoint.save(
            score=score,
            global_steps=global_steps,
            reload_metadata=False,
            **info,
        )

    def restore_checkpoint(self):
        SC = self.session_config
        restore_folder = SC.checkpoint.restore_folder
        if (restore_folder
            and U.f_last_part_in_path(restore_folder) != 'checkpoint'):
            # automatically append 'checkpoint' subfolder
            restore_folder = U.f_join(restore_folder, 'checkpoint')
        restored = self._periodic_checkpoint.restore(
            target=SC.checkpoint.learner.restore_target,
            mode=SC.checkpoint.learner.mode,
            reload_metadata=True,
            check_ckpt_exists=True,  # TODO set to False unless debugging
            restore_folder=restore_folder,
        )
        if restored:
            self.log.info('successfully restored from checkpoint', restored)

    ######
    # Main Loop
    # Override to completely change learner behavior
    ######
    def main_loop(self):    
        """
            Main loop that defines learner process
        """
        self.iter_timer.start()
        for i, batch in enumerate(self.fetch_iterator()):
            # if i % 20 == 0:
            #     cur_time = time.time()
            #     print('Iter {}'.format(i))
            #     print('Time {}'.format(cur_time - self.last_time_2))
            #     self.last_time_2 = cur_time
            self.current_iter = i
            data = batch.data
            with self.learn_timer.time():
                self.learn(data)
            with self.publish_timer.time():
                pass
                # self.publish_parameter(i, message='batch '+str(i))
            self.iter_timer.lap()
