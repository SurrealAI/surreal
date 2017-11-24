from .config import extend_config

# ======================== Agent-Learner side ========================
BASE_LEARN_CONFIG = {
    'model': '_dict_',
    'algo': '_dict_',
    'replay': {
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
        'local_batch_queue_size': '_int_',  # max batches to pre-sample
        'remote_exp_queue_size': '_int_',  # max size of Redis exp queue
        'local_exp_queue_size': '_int_',  # max number of exps to pre-fetch
        'tensorboard_display': True,  # display replay stats on Tensorboard
    },
    'sender': {
        'pointers_only': True,
        'remote_save_exp': False,  # store Exp tuples on Redis or not
        'local_obs_cache_size': '_int_',  # to avoid sending duplicate obs
    },
    'ps': {
        'name': 'ps',
        'host': '_str_',
        'port': '_int_',
    },
    'tensorplex': {
        'host': '_str_',
        'port': '_int_',
        'tensorboard_port': '_int_',  # tensorboard port
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
        'local_batch_queue_size': 10,  # max batches to pre-sample
        'remote_exp_queue_size': 10000,  # max size of Redis exp queue
        'local_exp_queue_size': 10000,  # max number of exps to pre-fetch
    },
    'sender': {
        'local_obs_cache_size': 10000,  # to avoid sending duplicate obs
    },
    'ps': {
        'host': 'localhost',
        'port': 6380,
    },
    'tensorplex': {
        'host': 'localhost',
        'port': 6381,
        'tensorboard_port': 6006,
        'interval_episodes': 10,
        'average_episodes': 10,
    },
}

LOCAL_SESSION_CONFIG = extend_config(LOCAL_SESSION_CONFIG, BASE_SESSION_CONFIG)
