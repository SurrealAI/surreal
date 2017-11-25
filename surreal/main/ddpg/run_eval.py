import surreal.utils as U
from surreal.agent.ddpg_agent import DDPGAgent
from surreal.env import *
from surreal.main.ddpg.configs import *

args = U.ArgParser()
args.add('mode', type=str)
args.add('--id', type=int, default=0)
args = args.parse()

agent_mode = AgentMode[args.mode]
assert agent_mode != AgentMode.training

if agent_mode == AgentMode.eval_deterministic:
    eval_id = 'deterministic'
else:
    eval_id = 'stochastic-{}'.format(args.id)

ddpg_agent = DDPGAgent(
    learn_config=learn_config,
    env_config=env_config,
    session_config=session_config,
    agent_id=eval_id,
    agent_mode=agent_mode,
)

env = gym.make('HalfCheetah-v1')
# env._max_episode_steps = 100
env = GymAdapter(env)
env = EvalTensorplexMonitor(
    env,
    eval_id=eval_id,
    pull_parameters=ddpg_agent.pull_parameters,
    session_config=session_config,
)

obs, info = env.reset()
while True:
    action = ddpg_agent.act(U.to_float_tensor(obs))
    obs, reward, done, info = env.step(action)
    if done:
        obs, info = env.reset()
