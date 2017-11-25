import sys
from surreal.agent.ddpg_agent import DDPGAgent
from surreal.env import *
from surreal.main.ddpg.configs import *
from surreal.session import *

if len(sys.argv) == 2:
    agent_id = int(sys.argv[1])
else:
    agent_id = 0
agent_mode = AgentMode.training

env = gym.make('HalfCheetah-v1')
env._max_episode_steps = 100
env = GymAdapter(env)

env = ExpSenderWrapper(
    env,
    session_config=session_config
)
if 1:
    env = TrainingTensorplexMonitor(
        env,
        agent_id=agent_id,
        session_config=session_config,
        separate_plots=True
    )
if 1:
    env = ConsoleMonitor(
        env,
        update_interval=10,
        average_over=10,
        extra_rows=OrderedDict()
    )

ddpg_agent = DDPGAgent(
    learn_config=learn_config,
    env_config=env_config,
    session_config=session_config,
    agent_id=agent_id,
    agent_mode=AgentMode.training
)


obs, info = env.reset()
while True:
    action = ddpg_agent.act(U.to_float_tensor(obs))
    obs, reward, done, info = env.step(action)
    if done:
        ddpg_agent.pull_parameters()
        obs, info = env.reset()
