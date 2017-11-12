import torch
from surreal.learner.dqn import DQNLearner
from surreal.replay import *
from surreal.main.cartpole_configs import *
from surreal.session import *


RedisClient().flushall()  # DEBUG ONLY


def debug_td_error(td_error):
    raw_loss = U.huber_loss_per_element(td_error)
    print(U.to_scalar(torch.mean(raw_loss)))


replay = UniformReplay(
    learn_config=cartpole_learn_config,
    env_config=cartpole_env_config,
    session_config=LOCAL_SESSION_CONFIG
)
dqn = DQNLearner(
    learn_config=cartpole_learn_config,
    env_config=cartpole_env_config,
    session_config=LOCAL_SESSION_CONFIG
)
for i, batch in enumerate(replay.sample_iterator()):
    td_error = dqn.learn(batch)
    debug_td_error(td_error)
    if (i+1) % 1 == 0:
        dqn.broadcast(message='batch '+str(i))

