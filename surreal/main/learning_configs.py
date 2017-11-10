cartpole_learning_config = {
    'model': {
        'input_shape': [4],
        'action_dim': 2,
        'convs': [],
        'fc_hidden_sizes': [64],
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
    'checkpoint': {
        'dir': '~/Train/cartpole',
        'freq': None,
    },
}

cartpole_env_config = {

}

session_config = {
    'redis': {
        'replay': {
            'host': 'localhost',
            'port': 6379,
        },
        'ps': {
            'host': 'localhost',
            'port': 6379,
        },
    },
    'log': {
        'file_name': None,
        'file_mode': 'w',
        'time_format': None,
        'print_level': 'INFO',
        'stream': 'out',
    },
}

# TODO temp workaround
cartpole_learning_config.update(session_config)
