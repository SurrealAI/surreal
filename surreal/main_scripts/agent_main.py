from surreal.env import *
from surreal.session import *
import surreal.utils as U
from surreal.agent import agentFactory

def agent_parser_setup(parser):
    parser.add_argument('id', type=int, help='agent id')

def run_agent_main(args, config):
    session_config, learner_config, env_config = config.session_config, config.learner_config, config.env_config
    agent_id = args.id

    env, env_config = make_env(env_config)

    env = ConsoleMonitor(
        env,
        update_interval=10,
        average_over=10,
        extra_rows=OrderedDict(
            # Exploration=show_exploration
        )
    )

    fetch_parameter_mode = session_config.agent.fetch_parameter_mode
    if fetch_parameter_mode.startswith('episode'):
        _fetch_mode = 'episode'
    elif fetch_parameter_mode.startswith('step'):
        _fetch_mode = 'step'
    else:
        raise ValueError('invalid pull_parameter_mode.')
    _fetch_interval = 1
    if ':' in fetch_parameter_mode:
        _, n = fetch_parameter_mode.split(':')
        _fetch_interval = int(n)

    agent_mode = AgentMode.training
    
    expSenderWrapper = expSenderWrapperFactory(learner_config.algo.experience)
    env = expSenderWrapper(env, learner_config, session_config)    
    env = TrainingTensorplexMonitor(
        env,
        agent_id=agent_id,
        session_config=session_config,
        separate_plots=True
    )

    agent_class = agentFactory(learner_config.algo.agent_class)
    agent = agent_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=agent_id,
        agent_mode=agent_mode,
    )

    pull_tracker = PeriodicTracker(_fetch_interval)
    obs, info = env.reset()
    while True:
        action = agent.act(U.to_float_tensor(obs))
        obs, reward, done, info = env.step(action)
        #time.sleep(0.1)
        if _fetch_mode == 'step' and pull_tracker.track_increment():
            is_fetched = agent.fetch_parameter()
        if done:
            obs, info = env.reset()
            if _fetch_mode == 'episode' and pull_tracker.track_increment():
                is_fetched = agent.fetch_parameter()
