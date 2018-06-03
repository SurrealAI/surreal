from surreal.env import *
from surreal.session import *
import surreal.utils as U
from surreal.agent import agent_factory
import argparse
import time
import numpy as np


def agent_parser_setup(parser):
    parser.add_argument('id', type=int, help='agent id')


def run_agent_main(args, config):
    np.random.seed(int(time.time() * 100000 % 100000))
    
    session_config, learner_config, env_config = \
        config.session_config, config.learner_config, config.env_config
    agent_id = args.id
    agent_mode = 'training'

    env, env_config = make_env(env_config, 'agent')

    agent_class = agent_factory(learner_config.algo.agent_class)
    agent = agent_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=agent_id,
        agent_mode=agent_mode,
    )

    agent.main_agent(env)

    ##########

    # # This has to go first as it alters step returns
    # limit_training_episode_length = learner_config.algo.limit_training_episode_length
    # if limit_training_episode_length > 0:
    #     env = MaxStepWrapper(env, limit_training_episode_length)

    # env = ConsoleMonitor(
    #     env,
    #     update_interval=10,
    #     average_over=10,
    #     extra_rows=OrderedDict(
    #         # Exploration=show_exploration
    #     )
    # )

    # fetch_parameter_mode = session_config.agent.fetch_parameter_mode
    # fetch_parameter_interval = session_config.agent.fetch_parameter_interval
    # if fetch_parameter_mode == 'episode':
    #     _fetch_mode = 'episode'
    # elif fetch_parameter_mode.startswith('step'):
    #     _fetch_mode = 'step'
    # else:
    #     raise ValueError('invalid fetch_parameter_mode: {}.'.format(fetch_parameter_mode))
    # _fetch_interval = fetch_parameter_interval

    # agent_mode = 'training'
    # expSenderWrapper = expSenderWrapperFactory(learner_config.algo.experience)
    # env = expSenderWrapper(env, learner_config, session_config)    
    # env = TrainingTensorplexMonitor(
    #     env,
    #     agent_id=agent_id,
    #     session_config=session_config,
    #     separate_plots=True
    # )

    # agent_class = agent_factory(learner_config.algo.agent_class)
    # agent = agent_class(
    #     learner_config=learner_config,
    #     env_config=env_config,
    #     session_config=session_config,
    #     agent_id=agent_id,
    #     agent_mode=agent_mode,
    # )

    # pull_tracker = PeriodicTracker(_fetch_interval)
    # while True:
    #     agent.pre_episode()
    #     obs, info = env.reset()
    #     while True:
    #         obs = U.to_float_tensor(obs)
    #         agent.pre_action(obs)
    #         action = agent.act(obs)
    #         obs_next, reward, done, info = env.step(action)
    #         agent.post_action(obs, action, obs_next, reward, done, info)
    #         obs = obs_next
    #         #time.sleep(0.1)
    #         if _fetch_mode == 'step' and pull_tracker.track_increment():
    #             is_fetched = agent.fetch_parameter()
    #         if done:
    #             break
    #     agent.post_episode()
    #     if _fetch_mode == 'episode' and pull_tracker.track_increment():
    #         is_fetched = agent.fetch_parameter()
