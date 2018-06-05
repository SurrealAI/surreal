import time
import numpy as np
from multiprocessing import Process
import surreal.utils as U
from surreal.env import *
from surreal.session import *
from surreal.agent import agent_factory


def eval_batch_parser_setup(parser):
    parser.add_argument('ids', type=str)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--render', action='store_true', default=False)

def run_eval_batch_main(args, config):
    evals = []
    eval_ids = [int(x) for x in args.ids.split(',')]
    for eval_id in eval_ids:
        agent = Process(target=_run_eval_main, args=[config, eval_id, args.mode, args.render])
        agent.start()
        evals.append(agent)
    for i, agent in enumerate(evals):
        agent.join()
        raise RuntimeError('Eval {} exited with code {}'.format(i, agent.exitcode))

def _run_eval_main(config, eval_id, mode, render):
    np.random.seed(int(time.time() * 100000 % 100000))

    session_config, learner_config, env_config = \
        config.session_config, config.learner_config, config.env_config

    env, env_config = make_env(env_config, 'eval')

    assert mode != 'training'

    agent_class = agent_factory(learner_config.algo.agent_class)
    agent = agent_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=eval_id,
        agent_mode=mode,
    )

    agent.main(env, render)
