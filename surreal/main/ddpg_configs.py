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
    parser.add_argument('--num-agents', type=int, required=True, help='number of agents used')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of GPUs to use, 0 for CPU only.')
    parser.add_argument('--agent-num-gpus', type=int, default=0,
                        help='number of GPUs to use for agent, 0 for CPU only.')

    args = parser.parse_args(args=argv)

    learner_config = {
        'model': {
            'convs': [],
            'actor_fc_hidden_sizes': [300, 200],
            'critic_fc_hidden_sizes': [400, 300],
            'use_layernorm': False,
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
            'experience': 'ExpSenderWrapperSSARNStepBootstrap',
            'use_z_filter': False,
            'gamma': .99,
            'n_step': 3,
            'stride': 1,
            'network': {
                'lr_actor': 1e-4,
                'lr_critic': 1e-3,
                'clip_actor_gradient': True,
                'actor_gradient_norm_clip': 1.,
                'clip_critic_gradient': False,
                'critic_gradient_norm_clip': 5.,
                'actor_regularization': 0.0,
                'critic_regularization': 0.0,
                'use_action_regularization': False,
                'use_double_critic': False,
                'target_update': {
                    #'type': 'soft',
                    #'tau': 1e-3,
                    'type': 'hard',
                    'interval': 500,
                },
            },
            'exploration': {
                'param_noise_type': None,
                'param_noise_sigma': 0.05,
                'param_noise_alpha': 1.15,
                'param_noise_target_stddev': 0.005,
                'noise_type': 'normal',
                # Agents will be uniformly distributed sigma values from 0.0 to max_sigma.  For example, with 3 agents
                # The sigma values will be 0.0, 0.33, 0.66
                'max_sigma': 1.0,
                #'noise_type': 'ou_noise',
                #'theta': 0.15,
                #'dt': 1e-3,
            },
        },
        'replay': {
            'replay_class': 'UniformReplay',
            'batch_size': 512,
            'memory_size': int(1000000/3), # Note that actual replay size is memory_size * replay_shards
            'sampling_start_size': 3000,
            'replay_shards': 3,
        },
        'parameter_publish': {
            # Minimum amount of time (seconds) between two parameter publish
            'min_publish_interval': 3,
        },
    }

    env_config = {
        'env_name': args.env,
        'num_agents': args.num_agents,
        'use_demonstration': False,
        'pixel_input': False,
        'use_grayscale': False,
        'action_repeat': 1,
        'frame_stacks': 3,
        'frame_stack_concatenate_on_agent': False,
        # Debug only: agent will sleep for this number of seconds between actions
        'sleep_time': 0.0,
        'limit_episode_length': 0, # 0 means no limit
        'video': {
            'record_video': False,
            'save_folder': None,
            'max_videos': 500,
            'record_every': 20,
        },
        'observation': {
            'pixel':['camera0', 'depth'],
            # if using ObservationConcatWrapper, low_dim inputs will be concatenated into 'flat_inputs'
            'low_dim':['position', 'velocity', 'proprio', 'cube_pos', 'cube_quat', 'gripper_to_cube', 'low-dim'],
            #'low_dim':['position', 'velocity', 'proprio'],
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
            'num_gpus': args.agent_num_gpus,
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
