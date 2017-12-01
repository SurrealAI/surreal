import sys
import inspect
from surreal.env import *
from surreal.session import *
import surreal.utils as U


def run_agent_main(*,
                   agent_class,
                   env,
                   learn_config,
                   env_config,
                   session_config,
                   fetch_parameter_mode='episode',
                   agent_extra_kwargs=None):
    """
    Args:
        fetch_parameter_mode: 'episode', 'episode:<n>', 'step', 'step:<n>'
            every episode, every n episodes, every step, every n steps
    """
    args = U.ArgParser()
    args.add('id', type=int)
    args = args.parse()
    agent_id = args.id

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

    env = ExpSenderWrapper(
        env,
        session_config=session_config
    )
    env = TrainingTensorplexMonitor(
        env,
        agent_id=agent_id,
        session_config=session_config,
        separate_plots=True
    )

    if agent_extra_kwargs is None:
        agent_extra_kwargs = {}

    agent = agent_class(
        learn_config=learn_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=agent_id,
        agent_mode=agent_mode,
        **agent_extra_kwargs
    )

    pull_tracker = PeriodicTracker(_fetch_interval)
    obs, info = env.reset()
    while True:
        action = agent.act(U.to_float_tensor(obs))
        obs, reward, done, info = env.step(action)
        if _fetch_mode == 'step' and pull_tracker.track_increment():
            is_fetched = agent.fetch_parameter()
        if done:
            obs, info = env.reset()
            if _fetch_mode == 'episode' and pull_tracker.track_increment():
                is_fetched = agent.fetch_parameter()


def run_eval_main(*,
                  agent_class,
                  env,
                  learn_config,
                  env_config,
                  session_config,
                  agent_extra_kwargs=None):
    assert inspect.isclass(agent_class)

    args = U.ArgParser()
    args.add('--mode', type=str, required=True)
    args.add('--id', type=int, default=0)
    args = args.parse()

    agent_mode = AgentMode[args.mode]
    assert agent_mode != AgentMode.training

    if agent_mode == AgentMode.eval_deterministic:
        eval_id = 'deterministic'
    else:
        eval_id = 'stochastic-{}'.format(args.id)

    if agent_extra_kwargs is None:
        agent_extra_kwargs = {}

    agent = agent_class(
        learn_config=learn_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=eval_id,
        agent_mode=agent_mode,
        **agent_extra_kwargs
    )

    env = EvalTensorplexMonitor(
        env,
        eval_id=eval_id,
        fetch_parameter=agent.fetch_parameter,
        session_config=session_config,
    )

    obs, info = env.reset()
    while True:
        action = agent.act(U.to_float_tensor(obs))
        obs, reward, done, info = env.step(action)
        if done:
            obs, info = env.reset()


def run_learner_main(*,
                     learner_class,
                     learn_config,
                     env_config,
                     session_config,
                     learner_extra_kwargs=None):
    assert inspect.isclass(learner_class)

    if learner_extra_kwargs is None:
        learner_extra_kwargs = {}

    learner = learner_class(
        learn_config=learn_config,
        env_config=env_config,
        session_config=session_config,
        **learner_extra_kwargs
    )
    for i, batch in enumerate(learner.fetch_iterator()):
        learner.learn(batch)
        learner.publish_parameter(i, message='batch '+str(i))


def run_replay_main(*,
                    replay_class,
                    learn_config,
                    env_config,
                    session_config,
                    replay_extra_kwargs=None):
    assert inspect.isclass(replay_class)
    if replay_extra_kwargs is None:
        replay_extra_kwargs = {}
    replay = replay_class(
        learn_config=learn_config,
        env_config=env_config,
        session_config=session_config,
        **replay_extra_kwargs
    )
    replay.start_threads()  # block forever

