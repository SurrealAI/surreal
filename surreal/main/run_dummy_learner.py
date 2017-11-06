from surreal.agents.q_agent import QAgent
from surreal.distributed.ps import *
from surreal.envs import *
from surreal.learners.dqn import DQN
from surreal.model.q_net import FFQfunc
from surreal.replay import *
from pprint import pprint

q_func = FFQfunc(
    input_shape=[4],
    action_dim=2,
    convs=[],
    fc_hidden_sizes=[64],
    dueling=False
)

client = RedisClient()
client.flushall()
broadcaster = TorchBroadcaster(client, debug=True)

client = RedisClient()
replay = DummyReplay(
    redis_client=client,
    # memory_size=100000,
    sampling_start_size=5,
    batch_size=7,
    fetch_queue_size=10,
)

replay.start_queue_threads()
replay.start_evict_thread(6, sleep_interval=4.)
for i, batch in replay.batch_iterator():
    print(batch['rewards'])
    print('='*30)
    # for item in replay._memory:
    #     pprint(item)
    # print('='*30)
    # input('...')

