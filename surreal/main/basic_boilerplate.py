import sys
import inspect
from surreal.env import *
from surreal.session import *
import surreal.utils as U
from surreal.agent import agentFactory
from surreal.learner import learnerFactory
from surreal.replay import replayFactory

def run_agent_main(*,
                   learner_config,
                   env_config,
                   session_config,
                   agent_extra_kwargs=None):
    args = U.ArgParser()
    # Ignored. For compatibility, to be fixed
    args.add('ignored', type=str)
    args.add('id', type=int)
    args = args.parse()
    agent_id = args.id

    env, env_config = make(env_config)

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

    if agent_extra_kwargs is None:
        agent_extra_kwargs = {}

    agent_class = agentFactory(learner_config.algo.agent_class)
    agent = agent_class(
        learner_config=learner_config,
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
        #time.sleep(0.1)
        if _fetch_mode == 'step' and pull_tracker.track_increment():
            is_fetched = agent.fetch_parameter()
        if done:
            obs, info = env.reset()
            if _fetch_mode == 'episode' and pull_tracker.track_increment():
                is_fetched = agent.fetch_parameter()


def run_eval_main(*,
                  learner_config,
                  env_config,
                  session_config,
                  agent_extra_kwargs=None):

    args = U.ArgParser()
    # Ignored. To be fixed
    args.add('ignored', type=str)
    args.add('--mode', type=str, required=True)
    args.add('--id', type=int, default=0)
    args.add('--render', action='store_true', default=False)
    args = args.parse()

    env, env_config = make(env_config)

    agent_mode = AgentMode[args.mode]
    assert agent_mode != AgentMode.training

    if agent_mode == AgentMode.eval_deterministic:
        eval_id = 'deterministic-{}'.format(args.id)
    else:
        eval_id = 'stochastic-{}'.format(args.id)

    if agent_extra_kwargs is None:
        agent_extra_kwargs = {}

    agent_class = agentFactory(learner_config.algo.agent_class)
    agent = agent_class(
        learner_config=learner_config,
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
        if args.render:
            env.render()
        # import numpy as np
        # action = np.random.randn(6)
        obs, reward, done, info = env.step(action)
        print('action', action)
        print('obs', obs)
        print('reward', reward)
        print('done', done)
        if done:
            obs, info = env.reset()


def run_learner_main(*,
                     learner_config,
                     env_config,
                     session_config,
                     learner_extra_kwargs=None):

    if learner_extra_kwargs is None:
        learner_extra_kwargs = {}

    env, env_config = make(env_config)
    del env # Does not work for dm_control as they don't clean up
    
    learner_class = learnerFactory(learner_config.algo.learner_class)
    learner = learner_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        **learner_extra_kwargs
    )
    for i, batch in enumerate(learner.fetch_iterator()):
        learner.learn(batch)
        learner.publish_parameter(i, message='batch '+str(i))


def run_replay_main(*,
                    learner_config,
                    env_config,
                    session_config,
                    replay_extra_kwargs=None):
    if replay_extra_kwargs is None:
        replay_extra_kwargs = {}

    env, env_config = make(env_config)
    del env

    replay_class = replayFactory(learner_config.replay.replay_class)
    replay = replay_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        **replay_extra_kwargs)
    replay.start_threads()  # block forever

