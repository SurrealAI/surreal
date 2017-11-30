import itertools
import threading
import time

import surreal.utils as U
from surreal.session import (Loggerplex, StatsTensorplex, Config, extend_config,
                             BASE_SESSION_CONFIG, BASE_ENV_CONFIG)
from .aggregator import torch_aggregate


class _EvictThread(U.StoppableThread):
    def __init__(self,
                 evict_func,
                 evict_args,
                 evict_kwargs,
                 update_interval=1.):
        """
        Args:
            evict_func: call evict from Replay object
            evict_args: passed to evict_func
            evict_kwargs: passed to evict_func
            update_interval:
        """
        super().__init__()
        self._evict_func = evict_func
        self._evict_args = evict_args
        self._evict_kwargs = evict_kwargs
        self._update_interval = update_interval

    def run(self):
        while True:
            if self.is_stopped():
                break
            self._evict_func(*self._evict_args, **self._evict_kwargs)
            time.sleep(self._update_interval)


class _TensorplexThread(U.StoppableThread):
    """
    Monitor the sizes of remote queue (LLEN of redis exp queue) and local fetch
    queue. The size will be displayed on Tensorboard as a percentage of max
    size. The increase or decrease of the queue size indicates the relative
    speed of consumer and producer processes, which can help us diagnose the
    system bottleneck.
    The thread is activated by session_config.replay.tensorboard_display=True
    """
    def __init__(self,
                 exp_queue,
                 session_config,
                 tensorplex,
                 update_interval=1.):
        super().__init__()
        self._exp_queue = exp_queue
        self._local_max_size = session_config.replay.local_exp_queue_size
        self._tensorplex = tensorplex
        self._remote_max_size = session_config.replay.remote_exp_queue_size
        self._update_interval = update_interval
        self._init_time = time.time()

    def run(self):
        while True:
            if self.is_stopped():
                break
            time.sleep(self._update_interval)
            local_percent = (1. * self._exp_queue.local_queue_size()
                             / self._local_max_size)
            remote_percent = (1. * self._exp_queue.remote_queue_size()
                             / self._remote_max_size)
            self._tensorplex.add_scalars({
                'local_exp_queue': local_percent,
                'remote_exp_queue': remote_percent
            }, global_step=int(time.time() - self._init_time))


class Replay(metaclass=U.AutoInitializeMeta):
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

        self.batch_size = self.replay_config.batch_size
        self._client = RedisClient(
            host=self.session_config.replay.host,
            port=self.session_config.replay.port
        )
        self.log = Loggerplex(
            name='replay',
            session_config=self.session_config
        )
        self.tensorplex = StatsTensorplex(
            section_name='replay',
            session_config=self.session_config
        )
        self._exp_queue = ExpQueue(
            redis_client=self._client,
            queue_name=self.session_config.replay.name,
            maxsize=self.session_config.replay.local_exp_queue_size,
        )
        self._batch_fetch_queue = BatchFetchQueue(
            redis_client=self._client,
            maxsize=self.session_config.replay.local_batch_queue_size,
        )
        self._evict_thread = None
        self._tensorplex_thread = None
        self._has_tensorplex = self.session_config.replay.tensorboard_display
        self._job_queue = U.JobQueue()

    def _initialize(self):
        self.start_queue_threads()

    def default_config(self):
        """
        Returns:
            dict of default configs, will be placed in learn_config['replay']
        """
        return {
            'batch_size': '_int_',
        }

    def _insert(self, exp_dict):
        """
        Add a new experience to the replay.
        Includes passive evict logic if memory capacity is exceeded.

        Args:
            exp_dict: experience dictionary with
                {"obs_pointers", "reward", "action", "info"} keys

        Returns:
            a list of exp_dict evicted, or empty list if still within capacity
        """
        raise NotImplementedError

    def _sample(self, batch_size):
        """
        This function is called in the `exp_fetch_queue` thread, its operation
        is async, i.e. overlaps with the insertion operations.

        Args:
            batch_size: passed from self.batch_size, defined in the
                constructor upfront.
            Note that `batch_size` is

        Returns:
            a list of exp_dicts
        """
        raise NotImplementedError

    def _evict(self, *args, **kwargs):
        """
        Actively evict old experiences.

        Returns:
            list of exp dicts that contain `exp_pointer` or `obs_pointers`.
            if the exp is not stored on Redis, they will be ignored.
        """
        return []

    def _start_sample_condition(self):
        """
        Tells the thread to start sampling only when this condition is met.
        For example, only when the replay memory has > 10K experiences.

        Returns:
            bool: whether to start sampling or not
        """
        raise NotImplementedError

    def _aggregate_batch(self, exp_list):
        """
        Will be called in `next_batch()` method to produce the actual inputs
        to the neural network training loop.

        Args:
            exp_list: list of experience dictionaries with actual observations
                {"obs", "reward", "action", "info"} keys

        Returns:
            batched Tensors, batched action/reward vectors, etc.
        """
        return torch_aggregate(
            exp_list,
            obs_spec=self.env_config.obs_spec,
            action_spec=self.env_config.action_spec,
        )

    def insert(self, exp_dict):
        """
        Must not sample and insert at the same time
        """
        evicted_exp_list = self._job_queue.process(
            self._insert,
            exp_dict
        )
        self._clean_evicted(evicted_exp_list)
        return evicted_exp_list

    def _wrapped_sample_before_fetch(self):
        """
        Returns:
            List of exp_dicts with obs_pointers, fed to the BatchFetchQueue
            None if start_sample_condition not met
        """
        if self._start_sample_condition():
            sampled_exp_list = self._sample(self.batch_size)
            U.assert_type(sampled_exp_list, list)
            # incr ref count so that it doesn't get evicted by insert()
            # make sure to decr count after fetch(obs_pointers)!!
            obs_pointers = []
            for exp in sampled_exp_list:
                if 'obs_pointers' in exp:
                    obs_pointers.extend(exp['obs_pointers'])
            incr_ref_count(self._client, obs_pointers)
            return sampled_exp_list
        else:
            return None

    def _sample_before_fetch(self):
        return self._job_queue.process(self._wrapped_sample_before_fetch)

    def _clean_evicted(self, evicted_exp_list):
        if not evicted_exp_list:
            return
        evict_exp_pointers = []
        evict_obs_pointers = []
        for exp in evicted_exp_list:
            U.assert_type(exp, dict)
            if 'exp_pointer' in exp:
                evict_exp_pointers.append(exp['exp_pointer'])
            if 'obs_pointers' in exp:
                obs_pointers = exp['obs_pointers']
                U.assert_type(obs_pointers, list)
                evict_obs_pointers.extend(obs_pointers)
        evict_obs_pointers = decr_ref_count(
            self._client,
            evict_obs_pointers,
            delete=False
        )
        # print('DEBUG deleted', evict_obs_pointers)
        # mass delete exp and obs (only when ref drop to 0) on Redis
        self._client.mdel(evict_obs_pointers + evict_exp_pointers)

    def _wrapped_evict(self, *args, **kwargs):
        evicted_exp_list = self._evict(*args, **kwargs)
        self._clean_evicted(evicted_exp_list)
        return evicted_exp_list

    def evict(self, *args, **kwargs):
        return self._job_queue.process(self._wrapped_evict, *args, **kwargs)

    def start_queue_threads(self):
        """
        Call this method to launch all background threads that talk to Redis.
        """
        self._job_queue.start_thread()
        self._exp_queue.start_enqueue_thread()
        self._exp_queue.start_dequeue_thread(self.insert)
        self._batch_fetch_queue.start_enqueue_thread(self._sample_before_fetch)
        if self._has_tensorplex:
            self.start_tensorplex_thread()

    def stop_queue_threads(self):
        self._job_queue.stop_thread()
        self._exp_queue.stop_enqueue_thread()
        self._exp_queue.stop_dequeue_thread()
        self._batch_fetch_queue.stop_enqueue_thread()
        if self._has_tensorplex:
            self.stop_tensorplex_thread()

    def start_evict_thread(self, *args, update_interval=1., **kwargs):
        if self._evict_thread is not None:
            raise RuntimeError('evict thread already running')
        self._evict_thread = _EvictThread(
            evict_func=self.evict,
            evict_args=args,
            evict_kwargs=kwargs,
            update_interval=update_interval
        )
        self._evict_thread.start()
        return self._evict_thread

    def stop_evict_thread(self):
        t = self._evict_thread
        t.stop()
        self._evict_thread = None
        return t

    def start_tensorplex_thread(self, update_interval=1.):
        if self._tensorplex_thread is not None:
            raise RuntimeError('tensorplex thread already running')
        self._tensorplex_thread = _TensorplexThread(
            exp_queue=self._exp_queue,
            session_config=self.session_config,
            tensorplex=self.tensorplex,
            update_interval=update_interval
        )
        self._tensorplex_thread.start()
        return self._tensorplex_thread

    def stop_tensorplex_thread(self):
        t = self._tensorplex_thread
        t.stop()
        self._tensorplex_thread = None
        return t

    def sample(self):
        exp_list = self._batch_fetch_queue.dequeue()
        return self._aggregate_batch(exp_list)

    def sample_iterator(self, stop_condition=None):
        """
        Args:
            stop_condition: function () -> bool
                if None, never stops.

        Yields:
            batched inputs to neural network
        """
        if stop_condition is None:
            stop_condition = lambda: True
        while stop_condition():
            yield self.sample()
