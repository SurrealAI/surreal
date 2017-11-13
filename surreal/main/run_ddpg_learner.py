import torch
from surreal.learner.ddpg import DDPGLearner
from surreal.replay import *
from surreal.main.halfcheetah_configs import *
from surreal.session import *

RedisClient().flushall()  # DEBUG ONLY

def debug_td_error(td_error):
    raw_loss = U.huber_loss_per_element(td_error)
    print(U.to_scalar(torch.mean(raw_loss)))


replay = UniformReplay(
    learn_config=halfcheetah_learn_config,
    env_config=halfcheetah_env_config,
    session_config=halfcheetah_session_config
)

ddpg = DDPGLearner(
    learn_config=halfcheetah_learn_config,
    env_config=halfcheetah_env_config,
    session_config=halfcheetah_session_config
)

for i, batch in enumerate(replay.sample_iterator()):
    td_error = ddpg.learn(batch)
    debug_td_error(td_error)
    if (i+1) % 1 == 0:
        ddpg.broadcast(message='batch '+str(i))

