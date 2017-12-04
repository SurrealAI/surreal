from surreal.main.ddpg_fifo.configs import *
from surreal.main.basic_boilerplate import run_replay_main
from surreal.replay import FIFOReplay


run_replay_main(
    replay_class=FIFOReplay,
    learn_config=learn_config,
    env_config=env_config,
    session_config=session_config,
)
