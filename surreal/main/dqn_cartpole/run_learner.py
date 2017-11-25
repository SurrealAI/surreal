import torch

from surreal.learner.dqn import DQNLearner
from surreal.main.dqn_cartpole.configs import *
from surreal.replay import *
from surreal.session import *

C = Config(cartpole_session_config)
for server in ['replay', 'ps', 'tensorplex']:
    RedisClient(
        host=C[server]['host'],
        port=C[server]['port']
    ).flushall()


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

tensorplex = dqn.tensorplex
for i, batch in enumerate(replay.sample_iterator()):
    td_error = dqn.learn(batch)
    if i % 20 == 0:
        mean_td_error = U.to_scalar(torch.mean(torch.abs(td_error)))
        tensorplex.add_scalar('mean_td_error', mean_td_error, i)
    dqn.push_parameters(i, message='batch '+str(i))

