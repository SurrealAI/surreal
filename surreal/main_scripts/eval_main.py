from surreal.env import *
from surreal.session import *
import surreal.utils as U
from surreal.agent import agent_factory
import time
import numpy as np


def eval_parser_setup(parser):
    parser.add_argument('id', type=int)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--render', action='store_true', default=False)


def run_eval_main(args, config):
    np.random.seed(int(time.time() * 100000 % 100000))

    session_config, learner_config, env_config = \
        config.session_config, config.learner_config, config.env_config

    env, env_config = make_env(env_config)

    agent_mode = args.mode
    assert agent_mode != 'training'

    agent_class = agent_factory(learner_config.algo.agent_class)
    agent = agent_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=args.id,
        agent_mode=agent_mode,
    )

    agent.main(env, args.render)