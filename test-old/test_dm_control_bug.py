from dm_control import suite
import numpy as np

env = suite.load(domain_name="cartpole", task_name="swingup")

action_spec = env.action_spec()

action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
time_step = env.step(action)
obs = time_step.observation['velocity']

print('obs', obs)

for i in range(10):
    time_step = env.step(action)

print('obs', obs)
print('obs_new', time_step.observation['velocity'])