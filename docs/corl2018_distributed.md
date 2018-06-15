# CORL 2018 Code Samples: Surreal Distributed RL

As promised in the paper, we will open-source the entire Surreal distributed learning infrastructure very soon. 

In this document, we will provide a quick overview of some of our API sample code for developing and evaluating new distributed RL algorithms. 

By the time of final release, we will add much more documentation and tutorials. 

Thank you so much for your time! 


## Actor class

The parallel actors interact with the simulated environments asynchronously to generate experience data. They send the data to a centralized Buffer server. 


```python
class Actor:
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 actor_id,
                 actor_mode):
        """
        Three major configs: 
        - learner
        - environment
        - session (distributed runtime).
        """
        self.learner_config = learner_config
        self.env_config = env_config
        self.session_config = session_config
        assert actor_mode in ACTOR_MODES
        self.actor_mode = actor_mode
        self.actor_id = actor_id

        self._setup_parameter_pull()
        self._setup_logging()

        self.current_episode = 0
        self.cumulative_steps = 0
        self.current_step = 0
        
    ###########################################################
    # Abstract methods to be overriden by RL algorithm authors
    ###########################################################
    
    def act(self, obs):
        """
        Abstract method: actor takes actions in the environment. 
        Different RL methods will write neural network forward pass here.
        You should check `self.actor_mode` in the function and change act()
        logic with respect to training VS evaluation.

        Args:
            obs: typically a single obs, make sure to vectorize it first before
                passing to the torch `model`.

        Returns:
            action to be executed in the env
        """
        raise NotImplementedError

    def module_dict(self):
        """
        Abstract method: the learnable parameters 
        
        Returns:
            a dict of {name -> surreal.utils.pytorch.Module}
        """
        raise NotImplementedError
    
    ###########################################################
    # Exposed public methods
    ###########################################################
    
    def fetch_parameter(self):
        """
        Extends base class fetch_parameters to add some logging
        """
        params, info = self._ps_client.fetch_parameter_with_info()
        if params:
            params = U.deserialize(params)
            params = self.on_parameter_fetched(params, info)
            self._module_dict.load(params)

    def fetch_parameter_info(self):
        """
        Fetch information about the parameters currently held by the parameter server
        """
        return self._ps_client.fetch_info()

    def set_actor_mode(self, actor_mode):
        """
        Args:
            actor_mode: 'training', 'eval_deterministic', or 'eval_stochastic'
        """
        assert actor_mode in ACTOR_MODES
        self.actor_mode = actor_mode
        
    ###########################################################
    # Advanced exposed methods
    # Override in subclass, NEED to call super().on_parameter_fetched() etc.
    # User need to take care of actor mode
    # For advanced usage
    ###########################################################
    
    def on_parameter_fetched(self, params, info):
        """
            Method called when a new parameter is fetched. Free to be inherited by subclasses.
        """
        # The time it takes for parameter to go from learner to actor
        if self.actor_mode == 'training':
            delay = time.time() - info['time']
            self.actions_per_param_update.add_value(self.actions_since_param_update)
            self.episodes_per_param_update.add_value(self.episodes_since_param_update)
            self.tensorplex.add_scalars({'.core/parameter_publish_delay_s': delay,
                        '.core/actions_per_param_update': self.actions_per_param_update.cur_value(),
                        '.core/episodes_per_param_update': self.episodes_per_param_update.cur_value()
                        })
            self.actions_since_param_update = 0
            self.episodes_since_param_update = 0
        return params

    def pre_action(self, obs):
        """
            Called before act is called by actor main script
        """
        if self.actor_mode == 'training':
            if self._fetch_parameter_mode == 'step' and \
                    self._fetch_parameter_tracker.track_increment():
                self.fetch_parameter()

    def post_action(self, obs, action, obs_next, reward, done, info):
        """
            Called after act is called by actor main script
        """
        self.current_step += 1
        self.cumulative_steps += 1
        if self.actor_mode == 'training':
            self.actions_since_param_update += 1
            if done:
                self.episodes_since_param_update += 1

    def pre_episode(self):
        """
            Called by actor process.
            Can beused to reset internal states before an episode starts
        """
        if self.actor_mode == 'training':
            if self._fetch_parameter_mode == 'episode' and \
                    self._fetch_parameter_tracker.track_increment():
                self.fetch_parameter()

    def post_episode(self):
        """
            Called by actor process.
            Can beused to reset internal states after an episode ends
            I.e. after the post_action when done = True
        """
        self.current_episode += 1

    ###########################################################
    # Main loops. 
    # Override this to fully customize the actor process
    ###########################################################
    
    def main(self, env, render=False):
        """
            Default Main loop
        Args:
            @env: the environment to run actor on
        """
        env = self.prepare_env(env)
        self.env = env
        self.fetch_parameter()
        while True:
            self.pre_episode()
            obs, info = env.reset()
            total_reward = 0.0
            while True:
                if render:
                    env.render()
                self.pre_action(obs)
                action = self.act(obs)
                obs_next, reward, done, info = env.step(action)
                total_reward += reward
                self.post_action(obs, action, obs_next, reward, done, info)
                obs = obs_next
                if done:
                    break
            self.post_episode()
            if self.current_episode % 20 == 0:
                print('episode', self.current_episode, 'reward', total_reward)

    def prepare_env(self, env):
        """
            Applies custom wrapper to the environment as necessary
        Returns:
            @env: The (possibly wrapped) environment
        """
        if self.actor_mode == 'training':
            return self.prepare_env_actor(env)
        else:
            return self.prepare_env_eval(env)

    def prepare_env_actor(self, env):
        """
            Applies custom wrapper to the environment as necessary
            Only changes actor behavior
        """
        # This has to go first as it alters step() return value
        limit_episode_length = self.env_config.limit_episode_length
        if limit_episode_length > 0:
            env = MaxStepWrapper(env, limit_episode_length)

        expSenderWrapper = expSenderWrapperFactory(self.learner_config.algo.experience)
        env = expSenderWrapper(env, self.learner_config, self.session_config)
        env = TrainingTensorplexMonitor(
            env,
            actor_id=self.actor_id,
            session_config=self.session_config,
            separate_plots=True
        )
        return env

    def prepare_env_eval(self, env):
        """
            Applies custom wrapper to the environment as necessary
            Only changes eval behavior
        """
        limit_episode_length = self.env_config.limit_episode_length
        if limit_episode_length > 0:
            env = MaxStepWrapper(env, limit_episode_length)

        env = EvalTensorplexMonitor(
            env,
            eval_id=self.actor_id,
            fetch_parameter=self.fetch_parameter,
            session_config=self.session_config,
        )

        env_category = self.env_config.env_name.split(':')[0]
        if self.env_config.video.record_video and self.actor_id == 0:
            if env_category == 'gym':
                env = GymMonitorWrapper(env, self.env_config, self.session_config)
            else:
                env = VideoWrapper(env, self.env_config, self.session_config)
        return env

    def main_actor(self, env):
        """
            Main loop ran by the actor script
            Override if you want to customize actor behavior completely
        """
        self.main(env)

    def main_eval(self, env):
        """
            Main loop ran by the eval script
            Override if you want to customize eval behavior completely
        """
        self.main(env)

    ###########################################################
    # Internal methods to help instantiate the actor
    ###########################################################
    
    def _initialize(self):
        """
        host and port specified by Symphony Orchestrator
        """
        host, port = os.environ['SYMPH_PS_FRONTEND_HOST'], os.environ['SYMPH_PS_FRONTEND_PORT']
        self._module_dict = self.module_dict()
        if not isinstance(self._module_dict, ModuleDict):
            self._module_dict = ModuleDict(self._module_dict)
        self._ps_client = ParameterClient(
            host=host,
            port=port,
        )
    
    def _setup_parameter_pull(self):
        self._fetch_parameter_mode = self.session_config.actor.fetch_parameter_mode
        self._fetch_parameter_interval = self.session_config.actor.fetch_parameter_interval
        self._fetch_parameter_tracker = PeriodicTracker(self._fetch_parameter_interval)

    def _setup_logging(self):
        """
            Creates tensorplex logger and loggerplex logger
            Initializes bookkeeping values
        """
        if self.actor_mode == 'training':
            logger_name = 'actor-{}'.format(self.actor_id)
            self.tensorplex = self._get_tensorplex(
                '{}/{}'.format('actor', self.actor_id))
        else:
            logger_name = 'eval-{}'.format(self.actor_id)
            self.tensorplex = self._get_tensorplex(
                '{}/{}'.format('eval', self.actor_id))

        self.log = get_loggerplex_client(logger_name, self.session_config)
        # record how long the current parameter have been used
        self.actions_since_param_update = 0
        self.episodes_since_param_update = 0
        # Weighted Average over ~100 parameter updates.
        self.actions_per_param_update = U.MovingAverageRecorder(decay=0.99)
        self.episodes_per_param_update = U.MovingAverageRecorder(decay=0.99)

    def _get_tensorplex(self, name):
        """
        Tensorplex is our distributed Tensorboard visualization tool. 
        It is not associated with Tensorflow any more and works with any numpy tensors. 
        
        Args:
            name: The name of the collection of metrics
        """
        tp = get_tensorplex_client(
            name,
            self.session_config
        )
        periodic_tp = PeriodicTensorplex(
            tensorplex=tp,
            period=self.session_config.tensorplex.update_schedule.actor,
            is_average=True,
            keep_full_history=False
        )
        return periodic_tp

```

## Buffer class

On-policy and off-policy deep RL methods employ two different mechanisms of consuming experiences for learning. We introduce a centralized buffer structure to support both. 

In the on-policy case, the buffer is a FIFO queue (`FIFOBuffer`) that hold experience tuples in a sequential ordering and discard experiences right after the model updates. 

In the off-policy case, the buffer becomes a fixed-size replay memory (`UniformBuffer`) that uniformly samples batches of data upon request to allow experience reusing. The buffer can be sharded on multiple nodes to increase networking capacity.

Both `FIFOBuffer` and `UniformBuffer` classes extend from the base class below. 


```python

class Buffer():
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 index=0):
        """
        Three major configs: 
        - learner
        - environment
        - session (distributed runtime).
        """
        self.learner_config = learner_config
        self.env_config = env_config
        self.session_config = session_config
        self.index = index

        collector_port = os.environ['SYMPH_COLLECTOR_BACKEND_PORT']
        sampler_port = os.environ['SYMPH_SAMPLER_BACKEND_PORT']
        self._collector_server = ExperienceCollectorServer(
            host='localhost',
            port=collector_port,
            # port=7001,
            exp_handler=self._insert_wrapper,
            load_balanced=True,
        )
        self._sampler_server = ZmqSimpleServer(
            host='localhost',
            port=sampler_port,
            handler=self._sample_request_handler,
            load_balanced=True,
        )
        self._evict_interval = self.session_config.buffer.evict_interval
        self._evict_thread = None
        self._setup_logging()

    ###########################################################
    # Abstract methods to be overriden by on-policy and off-policy buffers
    ###########################################################
    
    def insert(self, exp_dict):
        """
        Add a new experience to the buffer.
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
        For example, only when the buffer memory has > 10K experiences.

        Returns:
            bool: whether to start sampling or not
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns:
            Current buffer size
        """
        raise NotImplementedError

    ###########################################################
    # Publicly exposed methods 
    ###########################################################
    
    def start_threads(self):
        if self._has_tensorplex:
            self.start_tensorplex_thread()
        
        self._collector_server.start()
        
        if self._evict_interval:
            self.start_evict_thread()

        self._sampler_server.start()

    def join(self):
        self._collector_server.join()
        self._sampler_server.join()
        if self._has_tensorplex:
            self._tensorplex_thread.join()
        if self._evict_interval:
            self._evict_thread.join()

    def start_evict_thread(self):
        if self._evict_thread is not None:
            raise RuntimeError('evict thread already running')
        self._evict_thread = U.start_thread(self._evict_loop)
        return self._evict_thread

    def start_tensorplex_thread(self):
        if self._tensorplex_thread is not None:
            raise RuntimeError('tensorplex thread already running')
        self._tensorplex_thread = U.PeriodicWakeUpWorker(target=self.generate_tensorplex_report)
        self._tensorplex_thread.start()
        return self._tensorplex_thread

    ###########################################################
    # Internal methods to help instantiate the buffer
    ###########################################################
    
    def _setup_logging(self):
        self.log = get_loggerplex_client(
            '{}/{}'.format('buffer', self.index),
            self.session_config
        )
        self.tensorplex = get_tensorplex_client(
            '{}/{}'.format('buffer', self.index),
            self.session_config
        )
        self._tensorplex_thread = None
        self._has_tensorplex = self.session_config.buffer.tensorboard_display

        # Origin of all global steps
        self.init_time = time.time()
        # Number of experience collected by agents
        self.cumulative_collected_count = 0
        # Number of experience sampled by learner
        self.cumulative_sampled_count = 0
        # Number of sampling requests from the learner
        self.cumulative_request_count = 0
        # Timer for tensorplex reporting
        self.last_tensorplex_iter_time = time.time()
        # Last reported values used for speed computation
        self.last_experience_count = 0
        self.last_sample_count = 0
        self.last_request_count = 0
        self.insert_time = U.TimeRecorder(decay=0.99998)
        self.sample_time = U.TimeRecorder()
        self.serialize_time = U.TimeRecorder()
        # moving average of about 100s
        self.exp_in_speed = U.MovingAverageRecorder(decay=0.99)
        self.exp_out_speed = U.MovingAverageRecorder(decay=0.99)
        self.handle_sample_request_speed = U.MovingAverageRecorder(decay=0.99)

    def _insert_wrapper(self, exp):
        """
            Allows us to do some book keeping in the base class
        """
        self.cumulative_collected_count += 1
        with self.insert_time.time():
            self.insert(exp)

    def _sample_request_handler(self, req):
        """
        Handle requests to the learner
        https://stackoverflow.com/questions/29082268/python-time-sleep-vs-event-wait
        Since we don't have external notify, we'd better just use sleep
        """
        batch_size = U.deserialize(req)
        U.assert_type(batch_size, int)
        while not self.start_sample_condition():
            time.sleep(0.01)
        self.cumulative_sampled_count += batch_size
        self.cumulative_request_count += 1
        with self.sample_time.time():
            sample = self.sample(batch_size)
        with self.serialize_time.time():
            return U.serialize(sample)

    def _evict_loop(self):
        assert self._evict_interval
        while True:
            time.sleep(self._evict_interval)
            self.evict()
```

## Learner class

The learner continuously pulls batches of experiences from the buffer and performs algorithm-specific parameter updates. Because learning is centralized, it can take advantage of multi-GPU parallelism. Periodically, the learner posts the latest parameters to the parameter server, which then broadcasts to all actors to update their behavior policies.

Different RL algorithms PPO and DDPG should override the Learner class to perform parameter updates. 

```python

class Learner():
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config):
        """
        Three major configs: 
        - learner
        - environment
        - session (distributed runtime).
        """
        self.learner_config = learner_config
        self.env_config = env_config
        self.session_config = session_config
        self._setup_connection()
        self._setup_logging()
        self._setup_checkpoint()
        self._setup_batch_prefetch()

    ###########################################################
    # Abstract methods to be overriden by different learning algorithms
    ###########################################################
    
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

    def preprocess_batch(self, batch):
        '''
        Perform algorithm-specific preprocessing tasks, overridden in subclasses
        For example, ddpg converts relevant variables onto gpu
        '''
        return batch

    ###########################################################
    # Publicly exposed methods
    ###########################################################
    
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

    def fetch_batch(self):
        return self._prefetch_queue.get()

    def fetch_iterator(self):
        while True:
            yield self.fetch_batch()

    def start_tensorplex_thread(self):
        if self._tensorplex_thread is not None:
            raise RuntimeError('tensorplex thread already running')
        self._tensorplex_thread = U.PeriodicWakeUpWorker(target=self.generate_tensorplex_report)
        self._tensorplex_thread.start()
        return self._tensorplex_thread
        
    ######################
    # Main Loop
    # Override to completely change learner behavior
    ######################
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

    ######################
    # Checkpointing
    ######################
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

    ###########################################################
    # Abstract methods to be overriden by RL algorithm authors
    ###########################################################
    def _initialize(self):
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

    def _setup_connection(self):  
        # sampler_host = self.session_config.replay.sampler_frontend_host
        # sampler_port = self.session_config.replay.sampler_frontend_port
        ps_publish_port = os.environ['SYMPH_PARAMETER_PUBLISH_PORT']
        batch_size = self.learner_config.replay.batch_size
        # max_prefetch_batch_queue = self.session_config.learner.max_prefetch_batch_queue

        self._ps_publisher = None  # in _initialize()
        self._ps_port = ps_publish_port
        self._prefetch_queue = LearnerDataPrefetcher(
            session_config=self.session_config,
            batch_size=batch_size,
            preprocess_task = self._prefetch_thread_preprocess,
        )

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

    ######################
    # Batch Prefetch
    ######################
    def _prefetch_thread_preprocess(self, batch):
        batch = self.aggregator.aggregate(batch)
        return batch

    def _preprocess_batch(self):
        for batch in self.fetch_iterator():
            batch = BeneDict(batch.data)
            # The preprocess step creates Variables which will become GpuVariables
            batch = self.preprocess_batch(batch)
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

```

## Parameter Server

Periodically, parameter server receives the latest parameters from the Learner, 
then it broadcasts to all actors to update their behavior policies.

Parameter Server can be shared to increase throughput. 

```python
class ParameterServer(Process):
    """
    Standalone script for PS node that runs in an infinite loop.
    PS subscribes to upstream (learner) and REPs to downstream (agent)
    """
    def __init__(self,
                 publish_host,
                 publish_port,
                 serving_host,
                 serving_port,
                 load_balanced=False):
        """

        Args:
            publish_host: learner side publisher server
            publish_port:
            agent_port: PS server that responds to agent fetch_parameter requests
        """
        super().__init__()
        self.publish_host = publish_host
        self.publish_port = publish_port
        self.serving_host = serving_host
        self.serving_port = serving_port
        # self.serving_port = 7005
        self.load_balanced = load_balanced
        # storage
        self.parameters = None
        self.param_info = None

    def run(self):
        self._subscriber = ZmqSubClient(
            host=self.publish_host,
            port=self.publish_port,
            handler=self._set_storage,
            topic='ps',
            preprocess=U.deserialize,
        )
        self._server = ZmqSimpleServer(
            host=self.serving_host,
            port=self.serving_port,
            handler=self._handle_agent_request,
            preprocess=U.deserialize,
            postprocess=U.serialize,
            load_balanced=self.load_balanced,
        )
        self._subscriber.start()
        self._server.start()
        print('Parameter server started')
        self._subscriber.join()
        self._server.join()
        # print('Finished')
        # return 'abc'

    def _set_storage(self, data):
        self.parameters, self.param_info = data

    def _handle_agent_request(self, request):
        """
        Reply to agents pulling params

        Args:
            request: 3 types
             - "info": only info
             - "parameter:<last_hash>": returns None if hash is not changed
                since the last request
             - "both:<last_hash>": returns (None, info) if hash is not
                changed, otherwise (param, info)
        """
        if request == 'info':
            return self.param_info
        elif request.startswith('parameter'):
            if self.parameters is None:
                return None, ''
            _, last_hash = request.split(':', 1)
            current_hash = self.param_info['hash']
            if last_hash == current_hash:  # param not changed
                return None, current_hash
            else:
                return self.parameters, current_hash
        elif request.startswith('both'):
            if self.parameters is None:
                return None, None
            _, last_hash = request.split(':', 1)
            if last_hash == self.param_info['hash']:  # param not changed
                return None, self.param_info
            else:
                return self.parameters, self.param_info
        else:
            raise ValueError('invalid request: '+str(request))
```
