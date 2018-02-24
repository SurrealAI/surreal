from .config import extend_config

# ======================== Agent-Learner side ========================
BASE_LEARNER_CONFIG = {
    'model': '_dict_',
    'algo': {
        # Agent class to instantiate
        'agent_class': '_str_',
        # Learner class to instantiate
        'learner_class': '_str_',
        'experience': '_str_',
        'n_step': 1,
        'gamma': '_float_',
        'actor_regularization': 0.0,
        'critic_regularization': 0.0,
        'use_batchnorm': False,
        'limit_training_episode_length': 0,
        'agent_sleep_time': 0.0,
    },
    'replay': {
        # The replay class to instantiate
        'replay_class': '_str_',
        'batch_size': '_int_',
        'replay_shards': 1,
    }
}


# ======================== Env side ========================
BASE_ENV_CONFIG = {
    'env_name' : '_str_',
    'video' : {
        'record_video' : False,
        'max_videos': 10,
        'record_every': 10,
        'save_directory': '/mnt/snaps',
    },
    'action_spec': {},
    'obs_spec': {},
    # 'action_spec': {
    #     'dim': '_list_',
    #     'type': '_enum[continuous, discrete]_'
    # },
    # 'obs_spec': {
    #     'dim': '_list_',
    #     'type': ''  # TODO uint8 format
    # },
}


# ======================== Session side ========================
BASE_SESSION_CONFIG = {
    'folder': '_str_',

    'replay': {
        'collector_frontend_host': '_str_',  # upstream from agents' pusher
        'collector_frontend_port': '_int_',
        'collector_backend_host': '_str_',  # upstream from agents' pusher
        'collector_backend_port': '_int_',
        'sampler_frontend_host': '_str_',  # downstream to Learner request
        'sampler_frontend_port': '_int_',
        'sampler_backend_host': '_str_',  # downstream to Learner request
        'sampler_backend_port': '_int_',
        'max_puller_queue': '_int_',  # replay side: pull queue size
        'max_prefetch_batch_queue': '_int_',  # learner side: max number of batches to prefetch
        'evict_interval': '_float_',  # in seconds
        'tensorboard_display': True,  # display replay stats on Tensorboard
    },
    'sender': {
        'flush_iteration': '_int_',
        'flush_time': '_int_',
    },
    'ps': {
        'host': '_str_',  # downstream to agent requests
        'port': '_int_',
        'publish_host': '_str',  # upstream from learner
        'publish_port': '_int_'
    },
    'tensorplex': {
        'host': '_str_',
        'port': '_int_',
        'tensorboard_port': '_int_',  # tensorboard port
        'agent_bin_size': 8,
        'max_processes': 4,
        'update_schedule': {  # TODO rename this to 'periodic'
            # for TensorplexWrapper:
            'training_env': '_int_',  # env record every N episodes
            'eval_env': '_int_',
            'eval_env_sleep': '_int_',  # throttle eval by sleep n seconds
            # for manual updates:
            'agent': '_int_',  # agent.update_tensorplex()
            'learner': '_int_',  # learner.update_tensorplex()
        }
    },
    'loggerplex': {
        'host': '_str_',
        'port': '_int_',
        'overwrite': False,
        'level': 'info',
        'show_level': True,
        'time_format': 'hms',
        'enable_local_logger': '_bool_',
        'local_logger_level': 'info',
        'local_logger_time_format': 'hms'
    },
    'agent': {
        'fetch_parameter_mode': '_str_',
        'fetch_parameter_interval': int,
    },
    'learner': {
        'num_gpus': '_int_'
    },
    'checkpoint': {
        'restore': '_bool_',  # if False, ignore the other configs under 'restore'
        'restore_folder': None,  # if None, use the same session folder.
                            # Otherwise restore ckpt from another experiment dir.
        'learner': {
            'restore_target': '_int_',
            'mode': '_enum[best,history]_',
            'keep_history': '_int_',
            'keep_best': '_int_',
            'periodic': '_int_',
        },
        'agent': {
            'restore_target': '_int_',
            'mode': '_enum[best,history]_',
            'keep_history': '_int_',
            'keep_best': '_int_',
            'periodic': '_int_',
        },
    }
}


LOCAL_SESSION_CONFIG = {
    'folder': '_str_',

    'replay': {
        'collector_frontend_host': 'localhost',  # upstream from agents' pusher
        'collector_frontend_port': 7001,
        'collector_backend_host': 'localhost',  # upstream from agents' pusher
        'collector_backend_port': 7002,
        'sampler_frontend_host': 'localhost',  # downstream to Learner request
        'sampler_frontend_port': 7003,
        'sampler_backend_host': 'localhost',  # downstream to Learner request
        'sampler_backend_port': 7004,
        'max_puller_queue': 10000,  # replay side: pull queue size
        'max_prefetch_batch_queue': 10,  # learner side: max number of batches to prefetch
        'evict_interval': 0.,  # in seconds
        'tensorboard_display': True,  # display replay stats on Tensorboard
    },
    'sender': {
        'flush_iteration': '_int_',
        'flush_time': 0,
    },
    'ps': {
        'host': 'localhost',  # downstream to agent requests
        'port': 7005,
        'publish_host': 'localhost',  # upstream from learner
        'publish_port': 7006
    },
    'tensorplex': {
        'host': 'localhost',
        'port': 7007,
        'tensorboard_port': 6006,
        'update_schedule': { # TODO: rename this to 'periodic'
            # for TensorplexWrapper:
            'training_env': 20,  # env record every N episodes
            'eval_env': 20,
            'eval_env_sleep': 30,  # throttle eval by sleep n seconds
            # for manual updates:
            'agent': 20,  # agent.update_tensorplex()
            'learner': 20,  # learner.update_tensorplex()
        }
    },
    'loggerplex': {
        'host': 'localhost',
        'port': 7008,
        'enable_local_logger': True,
    },
    'agent': {
        # fetch_parameter_mode: 'episode', 'episode:<n>', 'step', 'step:<n>'
        # every episode, every n episodes, every step, every n steps
        'fetch_parameter_mode': 'episode',
        'fetch_parameter_interval': 1,
    },
    'learner': {
        'num_gpus': 0
    },
    'checkpoint': {
        'restore': False,  # if False, ignore the other configs under 'restore'
        'restore_folder': None,
        'learner': {
            'restore_target': 0,
            'mode': 'history',
            'keep_history': 2,
            'keep_best': 0, # TODO don't keep best unless we solve the learner score issue
            'periodic': 100,
        },
        'agent': {
            'restore_target': 0,
            'mode': 'history',
            'keep_history': 2,
            'keep_best': 0, # TODO don't keep best unless we solve the learner score issue
            'periodic': 100,
        },
    }
}

LOCAL_SESSION_CONFIG = extend_config(LOCAL_SESSION_CONFIG, BASE_SESSION_CONFIG)


KUBE_SESSION_CONFIG = {
    'folder': '_str_',

    'replay': {
        'collector_frontend_host': '_str_',  # upstream from agents' pusher
        'sampler_frontend_host': '_str_',  # downstream to Learner request
    },
    'sender': {
        'flush_iteration': '_int_',
    },
    'ps': {
        'host': '_str_',  # downstream to agent requests
        'publish_host': '_str_',  # upstream from learner
    },
    'tensorplex': {
        'host': '_str_',
    },
    'loggerplex': {
        'host': '_str_',
    },
}

KUBE_SESSION_CONFIG = extend_config(KUBE_SESSION_CONFIG, LOCAL_SESSION_CONFIG)
