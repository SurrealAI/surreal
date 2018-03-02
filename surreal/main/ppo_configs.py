from surreal.session import Config, LOCAL_SESSION_CONFIG
import argparse

# TODO：Documentation on config files

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
            'actor_gradient_clip_value': 5.,
            'clip_critic_gradient': True,
            'critic_gradient_clip_value': 5.,
            'gamma': .995,
            'lam': 0.97,
            'use_z_filter': True,
            'norm_adv': True,
            'init_log_sig': -1.,
            'n_step': 10,
            'trace_cutoff': 2.0,
            'is_weight_thresh': 2.5, 
            'is_weight_eps': 1e-3,
            'experience': 'ExpSenderWrapperMultiStepBehavePolicyMovingWindow',
            'stride': 1,
            'batch_size': 64, 
            # ppo specific parameters:
            'method': 'adapt',
            'lr_policy': 2e-4,
            'lr_baseline': 3e-4,
            'lr_scale_per_mil': -1.0, # scaling learning rate every 1 millions frames. -1 denote no annealing
            'epoch_policy': 5,
            'epoch_baseline': 5,
            'kl_targ': 0.01, # target KL divergence between before and after
            'kl_cutoff_coeff': 50, # penalty coeff when kl large
            'clip_epsilon_init': 0.2, # factor of clipped loss
            'beta_init': 1.0, # original beta
            'clip_range': (0.05, 0.3), # range of the adapted penalty factor
            'adj_thres': (0.5, 2.0), # threshold to magnify clip epsilon
            'beta_range': (1/35.0 , 35.0) # range of the adapted penalty factor
        },
        'replay': {
            'replay_class': 'FIFOReplay',
            'batch_size': 256,
            'memory_size': 512,
            'sampling_start_size': 256,
        },
        'eval': {
            'eps': 0.05  # 5% random action under eval_stochastic mode
        }
    }


    env_config = {
        'env_name': args.env,  
        'sleep_time': 1/350,
        'video': {
            'record_video': False,
            'save_directory': '/mnt/snaps/',
            'max_videos': 100,
            'record_every': 100,
        },
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
            }
        },
        'sender': {
            'flush_iteration': 3,
        },
        'learner': {
            'num_gpus': args.num_gpus,
        },
        'agent' : {
            'fetch_parameter_mode': 'step',
            'fetch_parameter_interval': 25,
        },
        'replay' : {
            'max_puller_queue': 3,
            'max_prefetch_batch_queue': 1,
        },
    })

    session_config.extend(LOCAL_SESSION_CONFIG)
    return learner_config, env_config, session_config
