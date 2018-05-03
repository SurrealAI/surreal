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
            'actor_fc_hidden_sizes': [300, 200],
            'critic_fc_hidden_sizes': [400, 300],
            'dueling': False,
            'conv_spec': {
                'out_channels': [64, 64],
                'kernel_sizes': [3, 5],
                'use_batch_norm': False
            },
            'mlp_spec': {
                'sizes': [128],
                'use_dropout': False
            },
        },
        'algo': {
            'agent_class': 'DDPGAgent',
            'learner_class': 'DDPGLearner',
            'lr_actor': 1e-4,
            'lr_critic': 1e-4,
            'optimizer': 'Adam',
            'clip_actor_gradient': True,
            'actor_gradient_clip_value': 1.,
            'clip_critic_gradient': False,
            'critic_gradient_clip_value': 5.,
            'gamma': .99,
            'target_update': {
                'type': 'soft',
                'tau': 1e-3,
                # 'type': 'hard',
                # 'interval': 100,
            },
            'use_z_filter': False,
            'exploration': {
                'noise_type': 'normal',
                # Assigns a sigma from the list to each agent. If only one agent, it uses default 0.3 sigma.
                # 5 agents works well. If you use more than 5 agents, the sigma values will wrap around.
                # For example, the sixth agent (with agent_id 5) will have sigma 0.3
                'sigma': [0.3, 0.0, 0.1, 0.2, 0.4],
                # 'noise_type': 'ou_noise',
                # 'theta': 0.15,
                # 'sigma': 0.3,
                # 'dt': 1e-3,
            },
            'actor_regularization': 0.0,
            'critic_regularization': 0.0,
            'use_batchnorm': False,
            'use_layernorm': True,
            # if input is uint8, algorithm will scale it down by a factor of 256.0
            'is_uint8_pixel_input': True,
            'limit_training_episode_length': 0, # 0 means no limit
            # 'agent_sleep_time': 1/50.0,
            #'agent_sleep_time': 1/10.0,
            'n_step': 5,
            'experience': 'ExpSenderWrapperSSARNStepBootstrap',
            'stride': 1,
        },
        'replay': {
            'replay_class': 'UniformReplay',
            'batch_size': 512,
            # 'memory_size': 1000000,
            'memory_size': int(1000000/3), # Note that actual replay size is memory_size * replay_shards
            'sampling_start_size': 3000,
            'replay_shards': 3,
        },
        'eval': {
            'eps': 0.05  # 5% random action under eval_stochastic mode
        },
        'parameter_publish': {
            # Minimum amount of time (seconds) between two parameter publish
            'min_publish_interval': 3, 
        },
    }

    env_config = {
        'env_name': args.env,
        'pixel_input': True,
        'frame_stacks': 3,
        'video': {
            'record_video': True,
            'save_folder': None,
            'max_videos': 500,
            'record_every': 100,
        },
        'observation': {
            'pixel':['pixels', 'image'],
            'low_dim':['flat_inputs', 'position', 'velocity', 'proprio'],
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
                'agent': 50,  # agent.tensorplex.add_scalars()
                'learner': 20,  # learner.tensorplex.add_scalars()
            }
        },
        'agent': {
            # fetch_parameter_mode: 'episode', 'episode:<n>', 'step', 'step:<n>'
            # every episode, every n episodes, every step, every n steps
            'fetch_parameter_mode': 'step',
            'fetch_parameter_interval': 200,
        },
        'sender': {
            'flush_iteration': 100,
        },
        'learner': {
            'prefetch_processes': 3,
            'num_gpus': args.num_gpus,
        },
    })

    session_config.extend(LOCAL_SESSION_CONFIG)
    return learner_config, env_config, session_config
