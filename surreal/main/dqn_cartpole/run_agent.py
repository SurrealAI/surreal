import sys
from surreal.agent.q_agent import QAgent
from surreal.env import *
from surreal.main.dqn_cartpole.configs import *
from surreal.main.basic_boilerplate import run_agent_main
from surreal.session import *

# def show_exploration(total_steps, num_episodes):
#     global q_agent
#     return str(int(100 * q_agent.exploration.value(total_steps)))+'%'

env = GymAdapter(gym.make('CartPole-v0'))
env = ConsoleMonitor(
    env,
    update_interval=10,
    average_over=10,
    extra_rows=OrderedDict(
        # Exploration=show_exploration
    )
)

run_agent_main(
    agent_class=QAgent,
    env=env,
    learn_config=learn_config,
    env_config=env_config,
    session_config=session_config,
    pull_parameter_mode='step',
)

