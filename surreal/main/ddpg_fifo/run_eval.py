import surreal.utils as U
from surreal.agent.ddpg_agent import DDPGAgent
from surreal.env import *
from surreal.main.ddpg_fifo.configs import *
from surreal.main.basic_boilerplate import run_eval_main


env = gym.make('HalfCheetah-v1')
# env._max_episode_steps = 100
env = GymAdapter(env)

env = ConsoleMonitor(
    env,
    update_interval=10,
    average_over=10,
)

run_eval_main(
    agent_class=DDPGAgent,
    env=env,
    learner_config=learner_config,
    env_config=env_config,
    session_config=session_config,
)

