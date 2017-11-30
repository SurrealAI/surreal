from scratch.utils import *
from scratch.dummy_env import *
from scratch.dummy_agent import *


sender = ExpSender(
    host='127.0.0.1',
    port=8001,
    flush_iteration=5
)

done = False
a = 0

ag = DummyAgent(0)
env = DummyEnv(ag.dummy_matrix, sleep=.01)


last_obs = ag.dummy_matrix
for i in range(100):
    a = i % 10
    obs, reward, done, info = env.step(a)
    info['td-error'] = reward/10.
    ret = sender.send([last_obs, obs], a, reward, done, info)
    if ret is not None:
        print(ret, i)
    last_obs = obs
