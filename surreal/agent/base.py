"""
A template class that defines base agent APIs
"""
import time
import os
import surreal.utils as U
from surreal.env import make_env
from surreal.session import (
    PeriodicTracker, PeriodicTensorplex,
    get_loggerplex_client, get_tensorplex_client,
)
from surreal.distributed import ParameterClient, ModuleDict
from surreal.env import (
    MaxStepWrapper,
    TrainingTensorplexMonitor,
    EvalTensorplexMonitor,
    VideoWrapper
)

AGENT_MODES = ['training', 'eval_deterministic', 'eval_stochastic']


class Agent(object, metaclass=U.AutoInitializeMeta):
    """
        Important: When extending this class, make sure to follow the init method signature so that 
        orchestrating functions can properly initialize custom agents.

        TODO: Extend the initilization to allow custom non-config per-agent settings.
            To be used to have a heterogeneous agent population
    """
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 agent_id,
                 agent_mode,
                 render=False):
        """
            Initialize the agent class, 
        """
        self.learner_config = learner_config
        self.env_config = env_config
        self.session_config = session_config

        assert agent_mode in AGENT_MODES
        self.agent_mode = agent_mode
        self.agent_id = agent_id

        self._setup_parameter_pull()
        self._setup_logging()

        self.current_episode = 0
        self.cumulative_steps = 0
        self.current_step = 0

        self.actions_since_param_update = 0
        self.episodes_since_param_update = 0

        self.render = render

    #######
    # Internal initialization methods
    #######
    def _initialize(self):
        """
            implements AutoInitializeMeta meta class.
            self.module_dict can only happen after the module is constructed by subclasses.
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
        self._fetch_parameter_mode = self.session_config.agent.fetch_parameter_mode
        self._fetch_parameter_interval = self.session_config.agent.fetch_parameter_interval
        self._fetch_parameter_tracker = PeriodicTracker(self._fetch_parameter_interval)

    def _setup_logging(self):
        """
            Creates tensorplex logger and loggerplex logger
            Initializes bookkeeping values
        """
        if self.agent_mode == 'training':
            logger_name = 'agent-{}'.format(self.agent_id)
            self.tensorplex = self._get_tensorplex(
                '{}/{}'.format('agent', self.agent_id))
        else:
            logger_name = 'eval-{}'.format(self.agent_id)
            self.tensorplex = self._get_tensorplex(
                '{}/{}'.format('eval', self.agent_id))

        self.log = get_loggerplex_client(logger_name, self.session_config)
        # record how long the current parameter have been used
        self.actions_since_param_update = 0
        self.episodes_since_param_update = 0
        # Weighted Average over ~100 parameter updates.
        self.actions_per_param_update = U.MovingAverageRecorder(decay=0.99)
        self.episodes_per_param_update = U.MovingAverageRecorder(decay=0.99)

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
        periodic_tp = PeriodicTensorplex(
            tensorplex=tp,
            period=self.session_config.tensorplex.update_schedule.agent,
            is_average=True,
            keep_full_history=False
        )
        return periodic_tp

    #######
    # Exposed abstract methods
    # Override in subclass, no need to call super().act etc.
    # Enough for basic usage
    #######
    def act(self, obs):
        """
        Abstract method for taking actions.
        You should check `self.agent_mode` in the function and change act()
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
        Returns:
            a dict of name -> surreal.utils.pytorch.Module
        """
        raise NotImplementedError

    #######
    # Advanced exposed methods
    # Override in subclass, NEED to call super().on_parameter_fetched() etc.
    # User need to take care of agent mode
    # For advanced usage
    #######
    def on_parameter_fetched(self, params, info):
        """
            Called when a new parameter is fetched.
        """
        if self.agent_mode == 'training':
            # The time it takes for parameter to go from learner to agent
            delay = time.time() - info['time']
            self.actions_per_param_update.add_value(
                self.actions_since_param_update)
            self.episodes_per_param_update.add_value(
                self.episodes_since_param_update)
            self.tensorplex.add_scalars(
                {
                    '.core/parameter_publish_delay_s': delay,
                    '.core/actions_per_param_update':
                        self.actions_per_param_update.cur_value(),
                    '.core/episodes_per_param_update':
                        self.episodes_per_param_update.cur_value()
                })
            self.actions_since_param_update = 0
            self.episodes_since_param_update = 0
        return params

    def pre_action(self, obs):
        """
            Called before act is called by agent main script
        """
        if self.agent_mode == 'training':
            if self._fetch_parameter_mode == 'step' and \
                    self._fetch_parameter_tracker.track_increment():
                self.fetch_parameter()

    def post_action(self, obs, action, obs_next, reward, done, info):
        """
            Called after act is called by agent main script
        """
        self.current_step += 1
        self.cumulative_steps += 1
        if self.agent_mode == 'training':
            self.actions_since_param_update += 1
            if done:
                self.episodes_since_param_update += 1

    def pre_episode(self):
        """
            Called by agent process.
            Can beused to reset internal states before an episode starts
        """
        if self.agent_mode == 'training':
            if self._fetch_parameter_mode == 'episode' and \
                    self._fetch_parameter_tracker.track_increment():
                self.fetch_parameter()

    def post_episode(self):
        """
            Called by agent process.
            Can beused to reset internal states after an episode ends
            I.e. after the post_action when done = True
        """
        self.current_episode += 1

    #######
    # Main loops.
    # Customize this to fully customize the agent process
    #######
    def main(self):
        """
            Default Main loop
        Args:
            @env: the environment to run agent on
        """
        self.main_setup()
        while True:
            self.main_loop()

    def main_setup(self):
        """
            Setup before constant looping
        """
        env = self.get_env()
        env = self.prepare_env(env)
        self.env = env
        self.fetch_parameter()

    def main_loop(self):
        """
            One loop of agent, runs one episode of the environment
        """
        env = self.env
        self.pre_episode()
        obs, info = env.reset()
        total_reward = 0.0
        while True:
            if self.render:
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
            self.log.info('Episode {} reward {}'
                          .format(self.current_episode,
                                  total_reward))

    def get_env(self):
        """
        Returns a subclass of EnvBase, created from self.env_config
        """
        if self.agent_mode in ['eval_deterministic', 'eval_stochastic']:
            env, _ = make_env(self.env_config, mode='eval')
        else:
            env, _ = make_env(self.env_config)
        return env

    def prepare_env(self, env):
        """
            Applies custom wrapper to the environment as necessary
        Args:
            @env: subclass of EnvBse

        Returns:
            @env: The (possibly wrapped) environment
        """
        if self.agent_mode == 'training':
            return self.prepare_env_agent(env)
        else:
            return self.prepare_env_eval(env)

    def prepare_env_agent(self, env):
        """
            Applies custom wrapper to the environment as necessary
            Only changes agent behavior
        """
        # This has to go first as it alters step() return value
        limit_episode_length = self.env_config.limit_episode_length
        if limit_episode_length > 0:
            env = MaxStepWrapper(env, limit_episode_length)
        env = TrainingTensorplexMonitor(
            env,
            agent_id=self.agent_id,
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
            eval_id=self.agent_id,
            fetch_parameter=self.fetch_parameter,
            session_config=self.session_config,
        )

        env_category = self.env_config.env_name.split(':')[0]
        if self.env_config.video.record_video and self.agent_id == 0:
            if env_category != 'gym':
                # gym video recording not supported due to bug in OpenAI gym
                # https://github.com/openai/gym/issues/1050
                env = VideoWrapper(env, self.env_config, self.session_config)
        return env

    def main_agent(self):
        """
            Main loop ran by the agent script
            Override if you want to customize agent behavior completely
        """
        self.main()

    def main_eval(self):
        """
            Main loop ran by the eval script
            Override if you want to customize eval behavior completely
        """
        self.main()

    #######
    # Exposed public methods
    #######
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

    def set_agent_mode(self, agent_mode):
        """
        Args:
            agent_mode: 'training', 'eval_deterministic', or 'eval_stochastic'
        """
        assert agent_mode in AGENT_MODES
        self.agent_mode = agent_mode
