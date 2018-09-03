from surreal.env.mujocomanip.default_env_configs import *
from surreal.env.mujocomanip.default_object_configs import *
from surreal.env.mujocomanip.mujocomanip_envs import *
import copy
import numpy as np

env_config = DEFAULT_PUSHER_CONFIG
env_config.display = True
object_config = XML_BALL_CONFIG
env_config.mujoco_object_spec = object_config
env = SurrealSawyerPushEnv(env_config)

obs,info = env.reset()
while True:
    obs,info = env.reset()

    ### TODO: we should implement 
    ### TODO: this might need clipping ###
    action = np.random.randn(8)
    # action[7] *= 0.020833
    for i in range(2000):
        action = np.random.randn(8) / 2
        action[7] = -1
        obs, reward, done, info = env.step(action)
        # 
        # obs, reward, done, info = env._step([0,-1,0,0,0,0,2])
        # print(obs, reward, done, info)
        env.render()
        if done:
            print('done: {}'.format(reward))
            break