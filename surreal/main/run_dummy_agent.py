import gym
from easydict import EasyDict

from scratch.dummy_agent import *
from scratch.dummy_env import *
from surreal.distributed.ps import *
from surreal.distributed import *
from surreal.model.q_net import FFQfunc
from surreal.replay import *
from surreal.session import *


client = RedisClient()
sender = ExpSender(
    client,
    'replay',
    pointers_only=True,
    save_exp_on_redis=True,
    obs_cache_size=5
)

ag = DummyAgent(0)
env = DummyEnv(ag.dummy_matrix, sleep=.02)

client = RedisClient()
# listener = TorchListener(client, q_func, q_agent.get_lock(), debug=True)
# listener.run_listener_thread()


last_obs = ag.dummy_matrix
for i in range(100):
    a = i % 10
    obs, reward, done, info = env.step(a)
    info['td-error'] = reward/10.
    sender.send([last_obs, obs], a, reward, done, info)
    last_obs = obs

