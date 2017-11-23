from surreal.session import Config, LOCAL_SESSION_CONFIG


cartpole_learn_config = {
    'model': {
        'convs': [],
        'fc_hidden_sizes': [128],
        'dueling': False
    },
    'algo': {
        'lr': 1e-3,
        # 'train_freq': 1,
        'optimizer': 'Adam',
        'grad_norm_clipping': 10,
        'gamma': .99,
        'target_network_update_freq': 250 * 64,
        'double_q': True,
        'exploration': {
            'schedule': 'linear',
            'steps': 30000,
            'final_eps': 0.01,
        },
        'prioritized': {
            'enabled': False,
            'alpha': 0.6,
            'beta0': 0.4,
            'beta_anneal_iters': None,
            'eps': 1e-6
        },
    },
    'replay': {
        'batch_size': 64,
        'memory_size': 100000,
        'sampling_start_size': 1000,
    },
    'sender': {
        'pointers_only': True,
        'save_exp_on_redis': False,
        'max_redis_queue_size': 100000,
        'obs_cache_size': 100000,
    }
}


cartpole_env_config = {
    'action_spec': {
        'dim': [2],
        'type': 'discrete'
    },
    'obs_spec': {
        'dim': [4],
    }
}


cartpole_session_config = Config({
    'folder': '~/Temp/cartpole',
    'tensorplex': {
        'tb_port': 6006,
        'average_episodes': 50,
        'interval_episodes': 50
    }
})

cartpole_session_config.extend(LOCAL_SESSION_CONFIG)
