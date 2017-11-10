import torch
from surreal.agent.q_agent import QAgent
from surreal.distributed.ps import *
from surreal.env import *
from surreal.learner.dqn import DQNLearner
from surreal.model.q_net import FFQfunc
from surreal.replay import *
from surreal.main.learning_configs import cartpole_learning_config

parser = U.ArgParser()
# parser.add('gpu', type=int)
parser.add('-s', '--save-dir', type=str, default='')
parser.add('-d', '--dueling', action='store_true')
parser.add('-r', '--prioritized', action='store_true')
args = parser.parse()


client = RedisClient()
# TODO debug only
client.flushall()

DEBUG = 0
replay = UniformReplay(
    redis_client=client,
    memory_size=100 if DEBUG else 100000,
    obs_spec={},
    action_spec={'type': 'discrete'},
    sampling_start_size=40 if DEBUG else 1000,
    batch_size=16 if DEBUG else 64,
    fetch_queue_size=5,
    exp_queue_size=100 if DEBUG else 10000
)


def debug_td_error(td_error):
    raw_loss = U.huber_loss_per_element(td_error)
    print(U.to_scalar(torch.mean(raw_loss)))


dqn = DQNLearner(cartpole_learning_config)
replay.start_queue_threads()
for i, batch in replay.batch_iterator():
    td_error = dqn.learn(batch)
    debug_td_error(td_error)
    if (i+1) % 1 == 0:
        dqn.broadcast(message='batch '+str(i))

