from surreal.learner.dqn import DQNLearner
from surreal.main.dqn_cartpole.configs import *
from surreal.main.basic_boilerplate import run_learner_main
from surreal.replay import UniformReplay


run_learner_main(
    learner_class=DQNLearner,
    replay_class=UniformReplay,
    learn_config=learn_config,
    env_config=env_config,
    session_config=session_config,
    clear_redis=True
)
