from surreal.utils.tmux import *
from surreal.comm import *
from surreal.replay import *
from scratch.dummy_agent import *
from scratch.dummy_env import *
from time import sleep

client = RedisClient()
replay = DummyReplay(client, 32, download_queue_size=5)
replay.start_threads()

for i, batch in replay.batch_iterator():
    print('NN pulling batch', i)
    print(batch['reward'], batch['info'])
    if i == 2:
        break
