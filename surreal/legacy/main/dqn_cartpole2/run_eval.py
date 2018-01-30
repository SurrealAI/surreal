import surreal.utils as U
from surreal.agent.q_agent import QAgent
from surreal.env import *
from surreal.main.dqn_cartpole.configs import *
from surreal.main.basic_boilerplate import run_eval_main


env = GymAdapter(gym.make('CartPole-v0'))

run_eval_main(
    agent_class=QAgent,
    env=env,
    learner_config=learner_config,
    env_config=env_config,
    session_config=session_config,
)

