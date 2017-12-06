from surreal.agent.ddpg_agent import DDPGAgent
from surreal.env import *
from surreal.main.basic_boilerplate import run_agent_main
from surreal.main.ddpg.configs import *

env = gym.make('HalfCheetah-v1')
env._max_episode_steps = 100
env = GymAdapter(env)

env = ConsoleMonitor(
    env,
    update_interval=10,
    average_over=10,
)

run_agent_main(
    agent_class=DDPGAgent,
    env=env,
    learn_config=learner_config,
    env_config=env_config,
    session_config=session_config,
    fetch_parameter_mode='episode',
)

