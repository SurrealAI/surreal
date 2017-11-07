import torch
from surreal.agent.q_agent import QAgent
from surreal.distributed.ps import *
from surreal.env import *
from surreal.learner.dqn import DQNLearner
from surreal.model.q_net import FFQfunc
from surreal.replay import *

parser = U.ArgParser()
# parser.add('gpu', type=int)
parser.add('-s', '--save-dir', type=str, default='')
parser.add('-d', '--dueling', action='store_true')
parser.add('-r', '--prioritized', action='store_true')
args = parser.parse()


CARTPOLE_CONFIG = {
    'lr': 1e-3,
    # 'train_freq': 1,
    'optimizer': 'Adam',
    'grad_norm_clipping': 10,
    'gamma': .99,
    'target_network_update_freq': 250 * 64,
    'double_q': True,
    'checkpoint': {
        'dir': '~/Train/cartpole' if not args.save_dir else args.save_dir,
        'freq': None,
    },
    'log': {
        'file_name': None,
        'file_mode': 'w',
        'time_format': None,
        'print_level': 'INFO',
        'stream': 'out',
    },
    'prioritized': {
        'enabled': args.prioritized,
        'alpha': 0.6,
        'beta0': 0.4,
        'beta_anneal_iters': None,
        'eps': 1e-6
    },
}

q_func = FFQfunc(
    input_shape=[4],
    action_dim=2,
    convs=[],
    fc_hidden_sizes=[64],
    dueling=False
)

client = RedisClient()
# TODO debug only
client.flushall()
broadcaster = TorchBroadcaster(client, debug=0)

client = RedisClient()

DEBUG = 0
replay = UniformReplay(
    redis_client=client,
    memory_size=100 if DEBUG else 100000,
    sampling_start_size=40 if DEBUG else 1000,
    batch_size=16 if DEBUG else 64,
    fetch_queue_size=5,
    exp_queue_size=100 if DEBUG else 100000
)

dqn = DQNLearner(
    config=CARTPOLE_CONFIG,
    model=q_func,
)

def debug_td_error(td_error):
    raw_loss = U.huber_loss_per_element(td_error)
    print(U.to_scalar(torch.mean(raw_loss)))


replay.start_queue_threads()
for i, batch in replay.batch_iterator():
    td_error = dqn.learn(batch, i)
    debug_td_error(td_error)
    if (i+1) % 1 == 0:
        broadcaster.broadcast(
            net=q_func,
            message='batch '+str(i)
        )

