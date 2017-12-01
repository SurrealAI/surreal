from surreal.learner.dqn import DQNLearner
from surreal.main.dqn_cartpole.configs import *
from surreal.main.basic_boilerplate import run_learner_main


run_learner_main(
    learner_class=DQNLearner,
    learn_config=learn_config,
    env_config=env_config,
    session_config=session_config,
)
