from surreal.env import *
from surreal.session import *
import surreal.utils as U
from surreal.learner import learnerFactory

def learner_parser_setup(parser):
    pass

def run_learner_main(args, config):
    session_config, learner_config, env_config = config.session_config, config.learner_config, config.env_config
    env, env_config = make_env(env_config)
    del env # Does not work for dm_control as they don't clean up
    
    learner_class = learnerFactory(learner_config.algo.learner_class)
    learner = learner_class(
        learner_config=learner_config,
        env_config=env_config,
        session_config=session_config)

    # for i, batch in enumerate(learner.fetch_iterator()):
    #     pass
    

    # for i, batch in enumerate(learner.fetch_iterator()):
    #     break
    # while True:
    #     learner.learn(batch)

    for i, batch in enumerate(learner.fetch_iterator()):
        learner.learn(batch)
        learner.publish_parameter(i, message='batch '+str(i))
