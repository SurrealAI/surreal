from surreal.session import Config, LOCAL_SESSION_CONFIG
import argparse

def generate(argv):
    """
    The function name must be `generate`.
    Will be called by `surreal.main_scripts.runner`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='name of the environment')
    parser.add_argument('--num-agents', type=int, required=True, help='number of agents used')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of GPUs to use, 0 for CPU only.')
    parser.add_argument('--agent-num-gpus', type=int, default=0,
                        help='number of GPUs to use for agent, 0 for CPU only.')


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
            'gamma': .995, 
            'n_step': 15, # 10 for without RNN
            'stride': 10, # 10 for without RNN
            'network': {
                'lr_actor': 1e-4,
                'lr_critic': 1e-4,
                'clip_actor_gradient': True,
                'actor_gradient_norm_clip': 1., 
                'clip_critic_gradient': True,
                'critic_gradient_norm_clip': 1.,
                'actor_regularization': 0.0,
                'critic_regularization': 0.0,
                'anneal':{  
                    'lr_scheduler': "LinearWithMinLR",
                    'frames_to_anneal': 5e6,
                    'lr_update_frequency': 100, 
                    'min_lr': 5e-5,
                },
            },
            # ppo specific parameters:
            'ppo_mode': 'adapt',
            'advantage':{
                'norm_adv': True,
                'lam': 0.97,
                'reward_scale': 0.01,
            },
            'rnn': {
                'if_rnn_policy': True, 
                'rnn_hidden': 100,
                'rnn_layer': 1,
                'horizon': 5,
            },
            'consts': {
                'init_log_sig': -2,
                'log_sig_range': 0.5,
                'epoch_policy': 10,
                'epoch_baseline': 10,
                'adjust_threshold': (0.5, 2.0), # threshold to magnify clip epsilon
                'kl_target': 0.01, # target KL divergence between before and after
            },
            'adapt_consts': {
                'kl_cutoff_coeff': 50, # penalty coeff when kl large
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
        'parameter_publish': {
            # Minimum amount of time (seconds) between two parameter publish
            'min_publish_interval': 0.2, 
            'exp_interval': 4096,  
        },
    }

    env_config = {
        'env_name': args.env, 
        'pixel_input': False,
        'frame_stacks': 3, 
        'sleep_time': 0,
        'video': {
            'record_video': False,
            'save_folder': None,
            'max_videos': 500,
            'record_every': 20,
        },
        'observation': {
            'pixel':['camera0'],
            'low_dim':['position', 'velocity', 'proprio', 'cube_pos', 'cube_quat', 'gripper_to_cube'],
        },
        'limit_episode_length': 0,
    }

    session_config = Config({
        'folder': '_str_',
        'tensorplex': {
            'update_schedule': {
                # for TensorplexWrapper:
                'training_env': 20,  # env record every N episodes
                'eval_env': 5,
                'eval_env_sleep': 2,  # throttle eval by sleep n seconds
                # for manual updates:
                'agent': 50,  # agent.update_tensorplex()
                'learner': 20,  # learner.update_tensorplex()
            },
        },
        'agent' : {
            'fetch_parameter_mode': 'step',
            'fetch_parameter_interval': 100, # 10 for without RNN
            'num_gpus': args.agent_num_gpus,
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
    Specific hyperparameters For Cheetah v. Hopper:
        * ADAPT
        * gamma: 0.995
        * lam: 0.97
        * kltarget: .01
        * update count: 10
        * kl_cutoff_coeff: 50
        * fetch_parameter_interval: 100
        * release_interval: 4096
        * zfilter: True/False
        * n_step: 30/15
        * stride: 20/10
        * initial learning rate: 3e-4/1e-4
        * annealed final learning rate: 5e-5
        * actor_gradient_norm_clip: 1./1.
        * critic_gradient_norm_clip: 5./1.
        * reward_scale: 1.0/0.01
        * init_log_sig: -1 / -2
        * log_sig_range: 0 / 0.5
'''