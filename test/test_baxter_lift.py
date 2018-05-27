import numpy as np
import imageio
from surreal.env import make_env
from surreal.session import Config

env_config = Config({
    'env_name': 'mujocomanip:BaxterLiftEnv',
    'pixel_input': True,
    'frame_stacks': 3,
    'sleep_time': 0.0,
    # 'limit_episode_length': 200, # 0 means no limit
    'limit_episode_length': 1000, # 0 means no limit
    'video': {
        'record_video': True,
        'save_folder': None,
        'max_videos': 500,
        'record_every': 100,
    },
    'observation': {
        'pixel':['camera0', 'depth'],
        # if using ObservationConcatWrapper, low_dim inputs will be concatenated into 'flat_inputs'
        # 'low_dim':['position', 'velocity', 'proprio', 'cube_pos', 'cube_quat', 'gripper_to_cube'],
        'low_dim':['position', 'velocity', 'proprio'],
    },
})

writer = imageio.get_writer('baxter_lift.mp4', fps=20)
env, env_config = make_env(env_config)

obs = env.reset()
for i in range(1000):
    action = np.ones(16)
    obs, reward, done, info = env.step(action)
    writer.append_data(obs['pixel']['camera0'].transpose(1,2,0))
writer.close()