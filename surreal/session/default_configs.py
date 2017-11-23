from .config import extend_config

# ======================== Agent-Learner side ========================
BASE_LEARN_CONFIG = {
    'model': '_dict_',
    'algo': '_dict_',
    'replay': {
        'name': 'replay',
        'batch_size': '_int_'
    }
}


# ======================== Env side ========================
BASE_ENV_CONFIG = {
    'action_spec': {
        'dim': '_list_',
        'type': '_enum[continuous, discrete]_'
    },
    'obs_spec': {
        'dim': '_list_',
        'type': ''  # TODO uint8 format
    }
}


# ======================== Session side ========================
BASE_SESSION_CONFIG = {
    'folder': '_str_',

    'replay': {
        'name': 'replay',
        'host': '_str_',
        'port': '_int_',
    },
    'ps': {
        'name': 'ps',
        'host': '_str_',
        'port': '_int_',
    },
    'tensorplex': {
        'host': '_str_',
        'port': '_int_',
        'tb_port': '_int_',  # tensorboard port
        'log_overwrite': False,
        'log_debug': False,
        'agent_bin_size': 8,
        'interval_episodes': '_int_',
        'average_episodes': '_int_',
    },
}


LOCAL_SESSION_CONFIG = {
    'folder': '_str_',

    'replay': {
        'host': 'localhost',
        'port': 6379,
    },
    'ps': {
        'host': 'localhost',
        'port': 6380,
    },
    'tensorplex': {
        'host': 'localhost',
        'port': 6381,
        'tb_port': 6006,
        'interval_episodes': 10,
        'average_episodes': 10,
    },
}

LOCAL_SESSION_CONFIG = extend_config(LOCAL_SESSION_CONFIG, BASE_SESSION_CONFIG)
