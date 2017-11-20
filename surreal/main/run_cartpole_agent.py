import gym
from tabulate import tabulate
from surreal.agent.base import AgentMode
from surreal.agent.q_agent import QAgent
from surreal.env import *
from surreal.replay import *
from surreal.session import *
from surreal.main.cartpole_configs import *


env = GymAdapter(gym.make('CartPole-v0'))
env = ExpSenderWrapper(
    env,
    learn_config=cartpole_learn_config,
    session_config=cartpole_session_config
)
env = EpisodeMonitor(env, filename=None)

q_agent = QAgent(
    learn_config=cartpole_learn_config,
    env_config=cartpole_env_config,
    session_config=cartpole_session_config,
    agent_id=1,
    agent_mode=AgentMode.training,
)

info_print = PeriodicTracker(100)

obs, info = env.reset()
for T in itertools.count():
    action = q_agent.act(U.to_float_tensor(obs))
    obs, reward, done, info = env.step(action)
    is_pulled = q_agent.pull_parameters()
    if done:
        # print(q_agent.pull_parameter_info())
        obs, info = env.reset()
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
