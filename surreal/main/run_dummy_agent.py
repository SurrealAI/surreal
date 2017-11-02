import surreal.utils as U
from surreal.comm import *
from surreal.replay import *
from surreal.ps import *
from surreal.envs import *
from surreal.session import *
from surreal.agents.q_agent import QAgent
from surreal.model.q_net import FFQfunc
from scratch.dummy_env import *
from scratch.dummy_agent import *
import gym
from tabulate import tabulate
from easydict import EasyDict

C = {
    'exploration': { # NOTE: piecewise schedule requires that fraction and final_eps
        # are lists of the same dim that specifies each "piece"
        'schedule': 'linear',
        'steps': 5000,
        'final_eps': 0.02,
    },
}
C = EasyDict(C)

client = RedisClient()
client.flushall()
sender = ExpSender(client, 'replay', obs_cache_size=5)

env = gym.make('CartPole-v0')
env = EpisodeMonitor(env, filename=None)

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
q_agent = DummyAgent(0)
env = DummyEnv(q_agent.dummy_matrix, sleep=.3)

client = RedisClient()
listener = TorchListener(client, q_func, q_agent.get_lock(), debug=True)
listener.run_listener_thread()


last_obs = q_agent.dummy_matrix
for i in range(100):
    a = i % 10
    obs, reward, done, info = env.step(a)
    info['td-error'] = reward/10.
    sender.send([last_obs, obs], a, reward, done, info)
    last_obs = obs

