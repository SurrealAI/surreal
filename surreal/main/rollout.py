import pickle
import sys
import time
import argparse
from os import path

from glob import glob

from surreal.env import *
import surreal.utils as U
from surreal.agent import PPOAgent

from benedict import BeneDict

def restore_model(folder):
    """
    Loads model from an experiment folder.
    """

    # choose latest ckpt
    max_iter = -1.
    max_ckpt = None
    for ckpt in glob(path.join(folder, "*.ckpt")):
        print(ckpt)
        iter_num = int(path.basename(ckpt).split('.')[1])
        if iter_num > max_iter:
            max_iter = iter_num
            max_ckpt = ckpt
    if max_ckpt is None:
        raise ValueError('No checkpoint available in folder {}'.format())
    path_to_ckpt = max_ckpt
    with open(path_to_ckpt, 'rb') as fp:
        data = pickle.load(fp)
    return data['model']

def restore_config(path_to_config):
    """
    Loads a config from a file.
    """
    configs = BeneDict.load_yaml_file(path_to_config)
    return configs

def restore_env(env_config):
    """
    Restores the environment.
    """
    #env_config.eval_mode.render = True
    env, env_config = make_env(env_config, 'eval')
    return env, env_config

def restore_agent(agent_class, learner_config, env_config, session_config, model):
    """
    Restores an agent from a model.
    """
    agent = agent_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config,
        agent_id=0,
        agent_mode='eval_deterministic_local',
    )
    agent.model.load_state_dict(model)
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,)
    parser.add_argument("--render", action='store_true',)
    args = parser.parse_args()
    folder = args.folder
    render = args.render

    # set a seed
    np.random.seed(int(time.time() * 100000 % 100000))

    # restore policy
    print("\nLoading policy located at {}\n".format(folder))
    model = restore_model(path.join(folder, 'checkpoint'))

    # restore the configs
    configs = restore_config(path.join(folder, 'config.yml'))
    session_config, learner_config, env_config = \
        configs.session_config, configs.learner_config, configs.env_config

    # session_config.agent.num_gpus = 0
    # session_config.learner.num_gpus = 0
    # env_config.env_name = 'mujocomanip:SawyerPegsRoundEnv'

    # restore the environment
    env_config.eval_mode.render = render
    env, env_config = restore_env(env_config)

    # restore the agent
    agent = restore_agent(PPOAgent, learner_config, env_config, session_config, model)
    print("Successfully loaded agent and model!")

    # do some rollouts
    while True:
        ob, info = env.reset()
        ret = 0.
        if render:
            env.unwrapped.viewer.viewer._hide_overlay = True
            env.unwrapped.viewer.set_camera(0)
        for i in range(200):
            a = agent.act(ob)
            ob, r, _, _ = env.step(a)
            print(ob)
            if render:
                # NOTE: we need to unwrap the environment here because some wrappers override render
                env.unwrapped.render()
            ret += r
        print("return: {}".format(ret))
