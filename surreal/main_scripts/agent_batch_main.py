import time
import numpy as np
from multiprocessing import Process
from surreal.env import *
from surreal.session import *
import surreal.utils as U
from surreal.agent import agent_factory
import argparse



def agent_batch_parser_setup(parser):
    parser.add_argument('ids', type=str, help='agent ids: e.g. 1,2,3,4')

def run_agent_batch_main(args, config):
    agents = []
    agent_ids = [int(x) for x in args.ids.split(',')]
    for agent_id in agent_ids:
        agent = Process(target=_run_agent_main, args=[config, agent_id])
        agent.start()
        agents.append(agent)
    for i, agent in enumerate(agents):
        agent.join()
        raise RuntimeError('Agent {} exited with code {}'.format(i, agent.exitcode))

def _run_agent_main(config, agent_id):
    np.random.seed(int(time.time() * 100000 % 100000))

    session_config, learner_config, env_config = \
        config.session_config, config.learner_config, config.env_config
    agent_id = agent_id
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
