import gym
from gym import wrappers

env = gym.make("HalfCheetah-v2")
env = wrappers.Monitor(env, "/tmp/gym")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()