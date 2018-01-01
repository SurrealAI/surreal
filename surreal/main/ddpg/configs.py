from surreal.session import Config, LOCAL_SESSION_CONFIG


learner_config = {
    'model': {
        'convs': [],
        'fc_hidden_sizes': [128],
        'dueling': False,
        'conv_spec': {
            'out_channels': [64, 64],
            'kernel_sizes': [3, 5],
            'use_batch_norm': False
        },
        'mlp_spec': {
            'sizes': [128],
            'use_dropout': False
        }
    },
    'algo': {
        'lr': 1e-3,
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
        'n_step': 5,
        'experience': 'ExpSenderWrapperSSARNStep',
    },
    'replay': {
        'batch_size': 256,
        'memory_size': 1000000,
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
        'dims': [17],
        'dim': [17],
    }
}


session_config = Config({
    'folder': '/tmp/halfcheetah',
    'tensorplex': {
        'update_schedule': {
            # for TensorplexWrapper:
            'training_env': 20,  # env record every N episodes
            'eval_env': 5,
            'eval_env_sleep': 30,  # throttle eval by sleep n seconds
            # for manual updates:
            'agent': 50,  # agent.update_tensorplex()
            'learner': 20,  # learner.update_tensorplex()
        }
    },
    'sender': {
        'flush_iteration': 100,
    }
})

session_config.extend(LOCAL_SESSION_CONFIG)
