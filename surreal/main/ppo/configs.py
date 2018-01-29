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
            'sizes': [64],
            'use_dropout': False
        }
    },
    'algo': {
        'agent_class': 'PPOAgent', 
        'learner_class': 'PPOLearner',
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
        'use_z_filter': False,
        'norm_adv': False,
        'exploration': {
            'noise_type': 'normal',
            'sigma': 0.37,
            # 'noise_type': 'ou_noise',
            # 'theta': 0.15,
            # 'sigma': 0.3,
            # 'dt': 5e-2,
        },
        'n_step': 10,
        'experience': 'ExpSenderWrapperMultiStepMovingWindow',
        'stride': 1,
        'batch_size': 64,
        # ppo specific parameters:
        'method': 'clip',
        'lam': 0.95, # GAE lambda
        'lr_policy': 1e-3,
        'lr_baseline': 1e-3,
        'lr_scale_per_mil': -1.0, # scaling learning rate every 1 millions frames. -1 denote no annealing
        'epoch_policy': 10,
        'epoch_baseline': 10,
        'kl_targ': 0.003, # target KL divergence between before and after
        'kl_cutoff_coeff': 50, # penalty coeff when kl large
        'clip_epsilon_init': 0.2, # factor of clipped loss
        'beta_init': 1.0, # original beta
        'clip_range': (0.05, 0.3), # range of the adapted penalty factor
        'adj_thres': (0.5, 2.0), # threshold to magnify clip epsilon
        'beta_range': (1/35.0 , 35.0) # range of the adapted penalty factor
    },
    'replay': {
        'replay_class': 'UniformReplay',
        'batch_size': 128,
        'memory_size': 30000,
        'sampling_start_size': 1000,
    },
    'eval': {
        'eps': 0.05  # 5% random action under eval_stochastic mode
    }
}


env_config = {
    'env_name': 'dm_control:cheetah-run',
}


session_config = Config({
    'folder': '~/Desktop/Research/tmp/halfcheetah',
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
