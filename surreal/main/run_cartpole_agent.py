import gym
from easydict import EasyDict
from tabulate import tabulate

from surreal.agent.base import AgentMode
from surreal.agent.q_agent import QAgent
from surreal.distributed import *
from surreal.distributed.ps import *
from surreal.env import *
from surreal.model.q_net import FFQfunc
from surreal.replay import *
from surreal.session import *
from surreal.main.learning_configs import cartpole_learning_config

client = RedisClient()
sender = ExpSender(
    client,
    'replay',
    obs_cache_size=5,
    pointers_only=True,
    save_exp_on_redis=False,
    max_redis_queue_size=200,
)

env = gym.make('CartPole-v0')
env = EpisodeMonitor(env, filename=None)

q_agent = QAgent(
    config=cartpole_learning_config,
    agent_mode=AgentMode.training,
)


info_print = PeriodicTracker(100)

obs = env.reset()
# q_agent.set_eval(stochastic=False)
for T in itertools.count():
    # print(binary_hash(q_agent.q_func.parameters_to_binary()))
    action = q_agent.act(U.to_float_tensor(obs))
    new_obs, reward, done, info = env.step(action)
    # replay_buffer.add(new_obs, action, reward, new_obs, float(done))
    sender.send([obs, new_obs], action, reward, done, info)
    obs = new_obs
    if done:
        obs = env.reset()
        if info_print.track_increment():
            # book-keeping, TODO should be done in Evaluator
            info_table = []
            avg_reward = np.mean(env.get_episode_rewards()[-10:])
            info_table.append(['Last 10 rewards', U.fformat(avg_reward, 3)])
            info_table.append(['Exploration',
                           str(int(100 * q_agent.exploration.value(T)))+'%'])
            avg_speed = 1 / (float(np.mean(env.get_episode_times()[-10:])) + 1e-6)
            info_table.append(['Speed iter/s', U.fformat(avg_speed, 1)])
            info_table.append(['Total steps', env.get_total_steps()])
            info_table.append(['Episodes', len(env.get_episode_rewards())])
            print(tabulate(info_table, tablefmt='fancy_grid'))
