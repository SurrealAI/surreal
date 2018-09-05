import platform
from osim.env import ProstheticsEnv

# only visualize on Mac
visualize = platform.system()=='Darwin'

env = ProstheticsEnv(visualize=visualize)
print('Import OK')
observation = env.reset()
for i in range(100):
    observation, reward, done, info = env.step(env.action_space.sample())
    print('r=', reward, 'done', done)
