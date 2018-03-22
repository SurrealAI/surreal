from surreal.env import *
from surreal.session import *
import surreal.utils as U
from surreal.replay import replay_factory, ShardedReplay

def replay_parser_setup(parser):
    pass

def run_replay_main(args, config):
    session_config, learner_config, env_config = config.session_config, config.learner_config, config.env_config
    env, env_config = make_env(env_config, session_config)
    del env

    sharded = ShardedReplay(learner_config=learner_config,
                            env_config=env_config,
                            session_config=session_config)

    sharded.launch()
    sharded.join()

    # replay_class = replay_factory(learner_config.replay.replay_class)
    # replay = replay_class(
    #     learner_config=learner_config,
    #     env_config=env_config,
    #     session_config=session_config)
    # replay.start_threads()  # block forever