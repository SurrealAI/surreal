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
    'tensorboard': {
        'host': '_str_',
        'port': '_int_',
        'folder': '_str_',
        'overwrite': False,
        'debug': False,
    },
}


LOCAL_SESSION_CONFIG = {
    'replay': {
        'host': 'localhost',
        'port': 6379,
    },
    'ps': {
        'host': 'localhost',
        'port': 6380,
    },
    'tensorboard': {
        'host': 'localhost',
        'port': 6381,
        'folder': '_str_',
    },
}

LOCAL_SESSION_CONFIG = extend_config(LOCAL_SESSION_CONFIG, BASE_SESSION_CONFIG)
