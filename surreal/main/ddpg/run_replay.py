from surreal.main.ddpg.configs import *
from surreal.main.basic_boilerplate import run_replay_main
from surreal.replay import UniformReplay


run_replay_main(
    replay_class=UniformReplay,
    learner_config=learner_config,
    env_config=env_config,
    session_config=session_config,
)
