from surreal.utils.tmux import *
from surreal.comm import *
from surreal.replay import *
from scratch.dummy_agent import *
from scratch.dummy_env import *


ag = DummyAgent(0)
ag.initialize()
env = DummyEnv(ag.dummy_matrix, sleep=.6)

client = RedisClient()
sender = ExpSender(client, 'replay')


last_obs = ag.dummy_matrix
for i in range(42):
    a = i % 10
    obs, reward, done, info = env.step(a)
    info['td-error'] = reward/10.
    sender.send([last_obs, obs], a, reward, done, info)
    last_obs = obs
