import gym
from easydict import EasyDict
from tabulate import tabulate

from surreal.agent.ddpg_agent import DDPGAgent
from surreal.distributed import *
from surreal.distributed.ps import *
from surreal.env import *
from surreal.model.ddpg_net import DDPGfunc
from surreal.replay import *
from surreal.session import *

client = RedisClient()
sender = ExpSender(
    client,
    'replay',
    obs_cache_size=5,
    pointers_only=True,
    save_exp_on_redis=False,
    max_redis_queue_size=200,
)

env = gym.make('HalfCheetah-v1')
env = EpisodeMonitor(env, allow_early_resets=True, filename=None)

ddpg_func = DDPGfunc(
    obs_dim=17,
    action_dim=6,
)

ddpg_agent = DDPGAgent(
    model=ddpg_func ,
    action_mode='train',
    action_dim=6,
)

client = RedisClient()
listener = TorchListener(client, ddpg_agent, debug=0)
listener.run_listener_thread()

info_print = PeriodicTracker(100)

obs = env.reset()
# q_agent.set_eval(stochastic=False)
for T in itertools.count():
    # print(binary_hash(q_agent.q_func.parameters_to_binary()))
    action = ddpg_agent.act(U.to_float_tensor(obs))
    env.render()
    new_obs, reward, done, info = env.step(action)

    # print('T = {} Reward = {:.4f} Done = {}'.format(T, reward, done))
    if (T+1) % 100 == 0: done = True

    sender.send([obs, new_obs], action, reward, done, info)
    obs = new_obs
    if done:
        obs = env.reset()
        if info_print.track_increment():
            # book-keeping, TODO should be done in Evaluator
            info_table = []
            avg_reward = np.mean(env.get_episode_rewards()[-10:])
            info_table.append(['Last 10 rewards', U.fformat(avg_reward, 3)])
            avg_speed = 1 / (float(np.mean(env.get_episode_times()[-10:])) + 1e-6)
            info_table.append(['Speed iter/s', U.fformat(avg_speed, 1)])
            info_table.append(['Total steps', env.get_total_steps()])
            info_table.append(['Episodes', len(env.get_episode_rewards())])
            print(tabulate(info_table, tablefmt='fancy_grid'))
