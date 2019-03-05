import pickle
import sys
import time
import argparse
from os import path

from glob import glob

from surreal.env import *
import surreal.utils as U
from surreal.agent import PPOAgent, DDPGAgent

from benedict import BeneDict

def restore_model(folder, filename):
    """
    Loads model from an experiment folder.
    """
    path_to_ckpt = path.join(folder, "checkpoint", filename)
    with open(path_to_ckpt, 'rb') as fp:
        data = pickle.load(fp)
    return data['model']

def restore_config(path_to_config):
    """
    Loads a config from a file.
    """
    configs = BeneDict.load_yaml_file(path_to_config)
    return configs

def restore_agent(agent_class, learner_config, env_config, session_config, render):
    """
    Restores an agent from a model.
    """
    agent = agent_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=0,
        agent_mode='eval_deterministic_local', # TODO: handle stochastic?
        render=render
    )
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--render", action='store_true',)
    parser.add_argument("--record", action='store_true',)
    parser.add_argument("--record-every", type=int,)
    parser.add_argument("--record-folder", type=str,)
    args = parser.parse_args()

    if args.record and args.record_folder is None:
        parser.error("--record requires --record-folder")

    folder = args.folder
    checkpoint = args.checkpoint
    render = args.render
    algo = PPOAgent if (not args.algo or args.algo != 'ddpg') else DDPGAgent
    record = args.record
    record_every = args.record_every if args.record_every else 1
    record_folder = args.record_folder

    # set a seed
    np.random.seed(int(time.time() * 100000 % 100000))

    # restore the configs
    configs = restore_config(path.join(folder, 'config.yml'))
    session_config, learner_config, env_config = \
        configs.session_config, configs.learner_config, configs.env_config

    model = restore_model(folder, checkpoint)

    # update the environment
    env_config.video.record_video = record
    env_config.video.record_every = record_every
    env_config.video.save_folder = record_folder
    env_config.eval_mode.render = render

    # restore the agent
    agent = restore_agent(algo, learner_config, env_config, session_config, render)
    agent.model.load_state_dict(model)

    agent.main()
