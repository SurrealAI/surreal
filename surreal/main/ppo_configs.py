from surreal.session import Config, LOCAL_SESSION_CONFIG
import argparse

def generate(argv):
    """
    The function name must be `generate`.
    Will be called by `surreal.main_scripts.runner`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='name of the environment')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of GPUs to use, 0 for CPU only.')

    args = parser.parse_args(args=argv)
    
    learner_config = {
        'model': {
            'convs':[], # this can wait until TorchX
            'actor_fc_hidden_sizes': [300, 200],
            'critic_fc_hidden_sizes': [300, 200],
            'cnn_feature_dim': 256,
            'use_layernorm': False,
        },
        'algo': {
            # base configs
            'agent_class': 'PPOAgent', 
            'learner_class': 'PPOLearner',
            'experience': 'ExpSenderWrapperMultiStepMovingWindowWithInfo',
            'use_z_filter': False,
            'use_r_filter': False,
            'gamma': .99, 
            'n_step': 30, # 10 for without RNN
            'stride': 20, # 10 for without RNN
            'network': {
                'lr_actor': 1e-4,
                'lr_critic': 1e-4,
                'clip_actor_gradient': True,
                'actor_gradient_norm_clip': 1., 
                'clip_critic_gradient': True,
                'critic_gradient_norm_clip': 5.,
                'actor_regularization': 0.0,
                'critic_regularization': 0.0,
                'anneal':{  
                    'lr_scheduler': "LinearWithMinLR",
                    'frames_to_anneal': 1e7,
                    'lr_update_frequency': 100, 
                    'min_lr': 1e-4,
                },
                'target_update':{
                    'type': 'hard',
                    'interval': 4096,
                },
            },
            # ppo specific parameters:
            'ppo_mode': 'adapt',
            'advantage':{
                'norm_adv': True,
                'lam': 1.0,
            },
            'rnn': {
                'if_rnn_policy': True, 
                'rnn_hidden': 100,
                'rnn_layer': 1,
                'horizon': 10,
            },
            'consts': {
                'init_log_sig': -1.5,
                'log_sig_range': 1,
                'is_weight_thresh': 2.5,
                'epoch_policy': 5,
                'epoch_baseline': 5,
                'adjust_threshold': (0.5, 2.0), # threshold to magnify clip epsilon
                'kl_target': 0.02, # target KL divergence between before and after
            },
            'adapt_consts': {
                'kl_cutoff_coeff': 500, # penalty coeff when kl large
                'beta_init': 1.0, # original beta
                'beta_range': (1/35.0 , 35.0), # range of the adapted penalty factor
                'scale_constant': 1.5,
            },
            'clip_consts': {
                'clip_epsilon_init': 0.2, # factor of clipped loss
                'clip_range': (0.05, 0.3), # range of the adapted penalty factor
                'scale_constant': 1.2,
            },

        },
        'replay': {
            'replay_class': 'FIFOReplay',
            'batch_size': 64,
            'memory_size': 96,
            'sampling_start_size': 64,
            'replay_shards': 1,
        },
    }

    env_config = {
        'env_name': args.env, 
        'pixel_input': True,
        'frame_stacks': 3, 
        'sleep_time': 0,
        'video': {
            'record_video': True,
            'save_folder': None,
            'max_videos': 500,
            'record_every': 100,
        },
        'observation': {
            'pixel':['camera0'],
            'low_dim':['position', 'velocity', 'proprio', 'cube_pos', 'cube_quat', 'gripper_to_cube'],
        },
        'limit_episode_length': 1000,
    }

    session_config = Config({
        'folder': '_str_',
        'tensorplex': {
            'update_schedule': {
                # for TensorplexWrapper:
                'training_env': 20,  # env record every N episodes
                'eval_env': 5,
                'eval_env_sleep': 30,  # throttle eval by sleep n seconds
                # for manual updates:
                'agent': 50,  # agent.update_tensorplex()
                'learner': 20,  # learner.update_tensorplex()
            },
        },
        'agent' : {
            'fetch_parameter_mode': 'step',
            'fetch_parameter_interval': 250, # 10 for without RNN
        },
        'sender': {
            'flush_iteration': 3,
        },
        'learner': {
            'num_gpus': args.num_gpus,
        },
        'replay' : {
            'max_puller_queue': 3,
            'max_prefetch_batch_queue': 1,
        },
    })

    session_config.extend(LOCAL_SESSION_CONFIG)
    return learner_config, env_config, session_config

'''
    Specific parameter without RNN difference:
        * n_step -> 10
        * stride -> 10
        * fetch_parameter_mode -> 'step'
        * fetch_parameter_interval -> 10
    Pixel specific parameter differnce:
        * param_release_min -> 8192 (instead of 4096)
'''