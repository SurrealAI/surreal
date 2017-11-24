import gym
import sys
from collections import OrderedDict
from surreal.agent.base import AgentMode
from surreal.agent.q_agent import QAgent
from surreal.env import *
from surreal.session import *
from surreal.main.cartpole_configs import *


if len(sys.argv) == 2:
    agent_id = int(sys.argv[1])
else:
    agent_id = 0
agent_mode = AgentMode.training

env = GymAdapter(gym.make('CartPole-v0'))
env = ExpSenderWrapper(
    env,
    session_config=cartpole_session_config
)
if 1:
    env = TensorplexMonitor(
        env,
        agent_id=agent_id,
        agent_mode=agent_mode,
        session_config=cartpole_session_config,
        separate_plots=True
    )
if 1:
    def show_exploration(total_steps, num_episodes):
        global q_agent
        return str(int(100 * q_agent.exploration.value(total_steps)))+'%'

    env = ConsoleMonitor(
        env,
        interval_episodes=cartpole_session_config.tensorplex.interval_episodes,
        average_episodes=cartpole_session_config.tensorplex.average_episodes,
        extra_rows=OrderedDict(
            Exploration=show_exploration
        )
    )

q_agent = QAgent(
    learn_config=cartpole_learn_config,
    env_config=cartpole_env_config,
    session_config=cartpole_session_config,
    agent_id=agent_id,
    agent_mode=agent_mode,
)


obs, info = env.reset()
while True:
    action = q_agent.act(U.to_float_tensor(obs))
    obs, reward, done, info = env.step(action)
    is_pulled = q_agent.pull_parameters()
    if done:
        obs, info = env.reset()
