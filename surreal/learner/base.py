"""
Template class for all learners
"""
import threading
import queue
import time
from benedict import BeneDict
import numpy as np
import surreal.utils as U
from surreal.session import (
    TimeThrottledTensorplex,
    get_loggerplex_client, get_tensorplex_client
)
from surreal.distributed import ParameterPublisher, LearnerDataPrefetcher


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
        self._setup_batch_prefetch()

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
            preprocess_task = self._prefetch_thread_preprocess,
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
        min_publish_interval = self.learner_config.parameter_publish.min_publish_interval
        self._ps_publish_tracker = U.TimedTracker(min_publish_interval)
        self._prefetch_queue.start()
        self.start_tensorplex_thread()
        # restore_checkpoint should be called _after_ subclass __init__
        # that's why we put it in _initialize()
        if self.session_config.checkpoint.restore:
            self.restore_checkpoint()

    ######
    # Parameter publish
    ######
    def should_publish_parameter(self):
        return self._ps_publish_tracker.track_increment()

    def publish_parameter(self, iteration, message=''):
        """
        Learner publishes latest parameters to the parameter server.

        Args:
            iteration: the current number of learning iterations
            message: optional message, must be pickleable.
        """
        self._ps_publisher.publish(iteration, message=message)

    ######
    # Getting data
    ######
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

        iter_per_s = iter_elapsed / time_elapsed
        # Number of iterations per second
        system_metrics['iter_per_s'] = iter_per_s
        # Number of experience bathces processed per second
        system_metrics['exp_per_s'] = iter_per_s * self.learner_config.replay.batch_size
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

        # These are system metrics, they don't add to counter or trigger updates
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
            min_interval=self.session_config.checkpoint.learner.min_interval,
            tracked_obj=self,
            tracked_attrs=tracked_attrs,
            keep_history=self.session_config.checkpoint.learner.keep_history,
            keep_best=self.session_config.checkpoint.learner.keep_best,
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
    # Batch Prefetch
    ######
    def preprocess(self, batch):
        '''
        Perform algorithm-specific preprocessing tasks, overridden in subclasses
        For example, ddpg converts relevant variables onto gpu
        '''
        return batch

    def _prefetch_thread_preprocess(self, batch):
        batch = self.aggregator.aggregate(batch)
        return batch

    def _preprocess_batch(self):
        for batch in self.fetch_iterator():
            batch = BeneDict(batch.data)
            # The preprocess step creates Variables which will become GpuVariables
            batch = self.preprocess(batch)
            self._preprocess_prefetch_queue.put(batch)

    def _setup_batch_prefetch(self):
        self._preprocess_prefetch_queue = queue.Queue(maxsize=2)
        self._preprocess_threads = []
        for i in range(1):
            self._preprocess_threads.append(threading.Thread(target=self._preprocess_batch))
            self._preprocess_threads[-1].start()

    def fetch_processed_batch_iterator(self):
        while True:
            yield self._preprocess_prefetch_queue.get()

    ######
    # Main Loop
    # Override to completely change learner behavior
    ######
    def main_loop(self):    
        """
            Main loop that defines learner process
        """
        self.iter_timer.start()
        self.publish_parameter(0, message='batch '+str(0))

        for i, data in enumerate(self.fetch_processed_batch_iterator()):
            self.current_iter = i
            with self.learn_timer.time():
                self.learn(data)
            if self.should_publish_parameter():
                with self.publish_timer.time():
                    # pass
                    self.publish_parameter(i, message='batch '+str(i))
            self.iter_timer.lap()
