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
    session_config=cartpole_session_config
)
dqn = DQNLearner(
    learn_config=cartpole_learn_config,
    env_config=cartpole_env_config,
    session_config=cartpole_session_config
)
for i, batch in enumerate(replay.sample_iterator()):
    td_error = dqn.learn(batch)
    debug_td_error(td_error)
    dqn.push_parameters(i, message='batch '+str(i))

