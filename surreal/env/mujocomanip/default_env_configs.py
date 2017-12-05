from surreal.session.config import extend_config

DEFAULT_BASE_MUJOCO_ENV_CONFIG = {
    'debug': False,
    'display': False,
    'control_freq': 100
}

DEFAULT_SAWYER_ENV_CONFIG = {
    'action_spec': {
        'dim': [8],
        'type': 'continuous'
    },
}
DEFAULT_SAWYER_ENV_CONFIG = extend_config(DEFAULT_SAWYER_ENV_CONFIG, DEFAULT_BASE_MUJOCO_ENV_CONFIG)

DEFAULT_SINGLE_OBJECT_TARGET_CONFIG = {
    'mujoco_object_spec': '_dict_',
    'table_size': (0.8, 0.8, 0.8),
    'min_target_xy_distance': (0.1,0.1),
    'table_friction': None,
    'reward_lose': -1,
    'reward_win': 1,
    'reward_action_norm_factor': 0,
    'reward_objective_factor': 5,
    'win_rel_tolerance': 1e-2,
    'gripper': '_str_',
    'obs_spec': {
        'dim': [37],
    }
}

DEFAULT_SINGLE_OBJECT_TARGET_CONFIG = extend_config(DEFAULT_SINGLE_OBJECT_TARGET_CONFIG, DEFAULT_SAWYER_ENV_CONFIG)

DEFAULT_PUSHER_CONFIG =  {
    'mujoco_object_spec': '_dict_',
    'min_target_xy_distance': (0.1,0.1),
    'reward_touch_object_factor':0.001,
    'reward_align_direction_factor':0.001,
    'gripper': 'PushingGripper',
}
DEFAULT_PUSHER_CONFIG = extend_config(DEFAULT_PUSHER_CONFIG, DEFAULT_SINGLE_OBJECT_TARGET_CONFIG)

DEFAULT_GRASPER_CONFIG =  {
    'mujoco_object_spec': {
        'type': 'box',
        'size': [0.02, 0.02, 0.02],
        'rgba': [1, 0, 0, 1],
    },
    'min_target_xy_distance': (0.1,0.1),
    'reward_touch_object_factor':0.001,
    'reward_align_direction_factor':0.001,
    'gripper': 'TwoFingerGripper',
}
DEFAULT_GRASPER_CONFIG = extend_config(DEFAULT_GRASPER_CONFIG, DEFAULT_SINGLE_OBJECT_TARGET_CONFIG)

DEFAULT_STACKER_CONFIG = {
    'mujoco_objects_spec': '_list_',
    'table_size': (0.8, 0.8, 0.8),
    'table_friction': None,
    'reward_lose': -2,
    'reward_win': 2,
    'reward_action_norm_factor': -0.1,
    'reward_objective_factor': 0.1,
    'win_rel_tolerance': 1e-2,
    'gripper': 'TwoFingerGripper',
    'obs_spec': {
        'dim': ['_int_'], # 9 * len(mujoco_objects_spec) + 28
    }
}
DEFAULT_STACKER_CONFIG = extend_config(DEFAULT_STACKER_CONFIG, DEFAULT_SAWYER_ENV_CONFIG)
