from surreal.session import Config, LOCAL_SESSION_CONFIG


learn_config = {
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
        'batch_size': 256,
        'memory_size': 100000,
        'sampling_start_size': 1000,
    },
    'eval': {
        'eps': 0.05  # 5% random action under eval_stochastic mode
    }
}


env_config = {
    'action_spec': {
        'dim': [6],
        'type': 'continuous'
    },
    'obs_spec': {
        'dim': [17],
    }
}


session_config = Config({
    'folder': '~/Temp/halfcheetah',
    'tensorplex': {
        'tensorboard_port': 6006,
        'agent_update_interval': 50,  # record every N episodes
        'eval_update_interval': 20,
    },
    'sender': {
        'pointers_only': True,
        'remote_save_exp': False,
        'local_obs_cache_size': 100000,
    }
})

session_config.extend(LOCAL_SESSION_CONFIG)
