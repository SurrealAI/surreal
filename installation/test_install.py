import sys
import numpy as np
import torch

print('testing entry cmd:', sys.argv)
assert len(sys.argv) == 4, 'entry point shlex quote error'

print('testing pytorch ...')
print(torch.FloatTensor([7, 8, 9]))


import glfw
print('testing glfw ...')
if glfw.init():
    print('GLFW success')
else:
    raise RuntimeError('GLFW init failure')


from dm_control import suite
print('testing dm_control ...')
env = suite.load(domain_name="cartpole", task_name="swingup")
print(env.action_spec())
print('render success:', env.physics.render().shape)
