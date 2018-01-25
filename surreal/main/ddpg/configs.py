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
        'clip_actor_gradient': True,
        'actor_gradient_clip_value': 1.,
        'gamma': .99,
        'target_update': {
            'type': 'soft',
            'tau': 1e-2,
            # 'type': 'hard',
            # 'interval': 100,
        },
        # 'target_network_update_freq': 250 * 64,
        'use_z_filter': False,
        'exploration': {
            'noise_type': 'normal',
            'sigma': 0.37,
            # 'noise_type': 'ou_noise',
            # 'theta': 0.15,
            # 'sigma': 0.3,
            # 'dt': 5e-2,
        },
        'n_step': 5,
        'experience': 'ExpSenderWrapperMultiStepMovingWindow',
        # 'experience': 'ExpSenderWrapperSSARNStepBoostrap',
        'stride': 1,
    },
    'replay': {
        'batch_size': 64,
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
