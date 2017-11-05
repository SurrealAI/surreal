from surreal.agents.q_agent import QAgent
from surreal.distributed.ps import *
from surreal.envs import *
from surreal.learners.dqn import DQN
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
    'target_network_update_freq': 500,
    'double_q': True,
    'checkpoint': {
        'dir': '~/Train/cartpole' if not args.save_dir else args.save_dir,
        'freq': None,
    },
    'log': {
        'freq': 100,
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

q_agent = QAgent(
    q_func=q_func,
    action_dim=2,
)

client = RedisClient()
broadcaster = TorchBroadcaster(client, debug=True)

client = RedisClient()
replay = TorchUniformReplay(
    redis_client=client,
    memory_size=100000,
    sampling_start_size=64,
    batch_size=32,
    download_queue_size=5,
)

dqn = DQN(
    config=CARTPOLE_CONFIG,
    agent=q_agent,
    replay=replay,
)

replay.start_threads()
for i, batch in replay.batch_iterator():
    print(batch)
    input('...')

    # dqn.train_batch(i, batch)
    # if (i+1) % 100 == 0:
    #     broadcaster.broadcast(
    #         net=q_func,
    #         message='batch '+str(i)
    #     )

