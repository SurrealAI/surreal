from surreal.env.mujocomanip.default_env_configs import *
from surreal.env.mujocomanip.default_object_configs import *
from surreal.env.mujocomanip.mujocomanip_envs import *
import copy
import numpy as np

from surreal.session import Config

env_config = DEFAULT_STACKER_CONFIG
env_config.display = True
object_configs = [ ]
for x in range(5):
    one_config = DEFAULT_RANDOM_BOX_CONFIG # This doesn't work
    one_config = Config(DEFAULT_RANDOM_BOX_CONFIG.to_dict())
    one_config.seed = np.random.randint(100000)
    object_configs.append(one_config)
env_config.mujoco_objects_spec = object_configs
env_config.obs_spec.dim = 9 * len(object_configs) + 28
env = SurrealSawyerStackEnv(env_config)

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