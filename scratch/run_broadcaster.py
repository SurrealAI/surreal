from scratch.utils import *
import gym

client = RedisClient()
q_func = FFQfunc(
    input_shape=[4],
    action_dim=2,
    convs=[],
    fc_hidden_sizes=[64],
    dueling=False
)
q_func.load('cartpole.ckpt')
print(binary_hash(q_func.parameters_to_binary()))
client.flushall()

notifier = TorchBroadcaster(client)
notifier.broadcast(q_func, 'NEW PARAMS ARRIVE!')

env = gym.make('CartPole-v0')
env = EpisodeMonitor(env, filename=None)

q_agent = QAgent(
    q_func=q_func,
    action_dim=2,
)
q_agent.set_eval(stochastic=False)
obs = env.reset()

sys.exit()

while True:
    # Take action and update exploration to the newest value
    # myarray[None] unsqueeze a new dim
    action = q_agent.act(U.to_float_tensor(obs), vectorize=True)
    new_obs, reward, done, info = env.step(action)
    obs = new_obs
    if done:
        obs = env.reset()
        print(info)
        sleep(3)
