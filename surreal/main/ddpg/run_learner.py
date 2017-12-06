from surreal.learner.ddpg import DDPGLearner
from surreal.main.ddpg.configs import *
from surreal.replay import *
from surreal.main.basic_boilerplate import run_learner_main


run_learner_main(
    learner_class=DDPGLearner,
    learn_config=learner_config,
    env_config=env_config,
    session_config=session_config,
)
