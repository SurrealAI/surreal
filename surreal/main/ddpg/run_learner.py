import torch
from surreal.learner.ddpg import DDPGLearner
from surreal.main.ddpg.configs import *
from surreal.replay import *
from surreal.session import *

C = Config(session_config)
for server in ['replay', 'ps', 'tensorplex']:
    RedisClient(
        host=C[server]['host'],
        port=C[server]['port']
    ).flushall()


replay = UniformReplay(
    learn_config=learn_config,
    env_config=env_config,
    session_config=session_config
)
learner = DDPGLearner(
    learn_config=learn_config,
    env_config=env_config,
    session_config=session_config
)

tensorplex = learner.tensorplex
for i, batch in enumerate(replay.sample_iterator()):
    learner.learn(batch)
    learner.push_parameters(i, message='batch '+str(i))

