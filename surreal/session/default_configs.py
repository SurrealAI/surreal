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
        'host': '_str_',
        'port': '_int_',
        'local_batch_queue_size': '_int_',  # max batches to pre-sample
        'remote_exp_queue_size': '_int_',  # max size of Redis exp queue
        'local_exp_queue_size': '_int_',  # max number of exps to pre-fetch
        'tensorboard_display': True,  # display replay stats on Tensorboard
    },
    'sender': {
        'flush_iteration': '_int_',
        'flush_time': '_int_',
    },
    'ps': {
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
        'update_schedule': {
            # for TensorplexWrapper:
            'training_env': '_int_',  # env record every N episodes
            'eval_env': '_int_',
            'eval_env_sleep': '_int_',  # throttle eval by sleep n seconds
            # for manual updates:
            'agent': '_int_',  # agent.update_tensorplex()
            'learner': '_int_',  # learner.update_tensorplex()
        }
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
        'flush_iteration': '_int_',
        'flush_time': 0,
    },
    'ps': {
        'host': 'localhost',
        'port': 6380,
    },
    'tensorplex': {
        'host': 'localhost',
        'port': 6381,
        'tensorboard_port': 6006,
        'update_schedule': {
            # for TensorplexWrapper:
            'training_env': 20,  # env record every N episodes
            'eval_env': 20,
            'eval_env_sleep': 30,  # throttle eval by sleep n seconds
            # for manual updates:
            'agent': 20,  # agent.update_tensorplex()
            'learner': 20,  # learner.update_tensorplex()
        }
    },
}

LOCAL_SESSION_CONFIG = extend_config(LOCAL_SESSION_CONFIG, BASE_SESSION_CONFIG)
