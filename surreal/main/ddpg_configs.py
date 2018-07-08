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
            'conv_spec': {
                # First conv layer: 16 out channels, second layer 32 channels
                'out_channels': [16, 32],
                # First conv layer: kernel size 8, second layer kernel size 4
                'kernel_sizes': [8, 4],
                # First conv layer: stride=4, second layer stride=2
                'strides': [4, 2],
                # After final convolution, reshapes output to flat tensor and feed through mlp with output of this size
                'hidden_output_dim': 200,
            },
        },
        'algo': {
            'agent_class': 'DDPGAgent',
            'learner_class': 'DDPGLearner',
            'experience': 'ExpSenderWrapperSSARNStepBootstrap',
            'gamma': .99,
            # Unroll the bellman update
            'n_step': 3,
            # Send experiences every `stride` steps
            'stride': 1,
            'network': {
                'lr_actor': 1e-4,
                'lr_critic': 1e-3,
                'clip_actor_gradient': True,
                'actor_gradient_value_clip': 1.,
                'clip_critic_gradient': False,
                'critic_gradient_value_clip': 5.,
                # Weight regularization
                'actor_regularization': 0.0,
                'critic_regularization': 0.0,
                # beta version: see https://arxiv.org/pdf/1802.09477.pdf and
                # https://github.com/sfujim/TD3/blob/master/TD3.py
                # for action regularization and double critic algorithm details
                'use_action_regularization': False,
                'use_double_critic': False,
                'target_update': {
                    # Soft: after every iteration, target_params = (1 - tau) * target_params + tau * params
                    #'type': 'soft',
                    #'tau': 1e-3,
                    # Hard: after `interval` iterations, target_params = params
                    'type': 'hard',
                    'interval': 500,
                },
            },
            'exploration': {
                # Beta implementation of parameter noise:
                # see https://blog.openai.com/better-exploration-with-parameter-noise/ for algorithm details
                'param_noise_type': None,

                # normal parameter noise applies gaussian noise over the agent's parameters
                # 'param_noise_type': 'normal',

                # adaptive parameter noise scales the noise sigma up or down in order to achieve the target action
                # standard deviation
                # 'param_noise_type': 'adaptive_normal',
                'param_noise_sigma': 0.05,
                'param_noise_alpha': 1.15,
                'param_noise_target_stddev': 0.005,

                # Vanilla noise: applies gaussian noise on every action
                'noise_type': 'normal',
                # Agents will be uniformly distributed sigma values from 0.0 to max_sigma.  For example, with 3 agents
                # The sigma values will be 0.0, 0.33, 0.66
                'max_sigma': 1.0,

                # Or, use Ornstein-Uhlenbeck noise instead of gaussian
                #'noise_type': 'ou_noise',
                #'theta': 0.15,
                #'dt': 1e-3,
            },
        },
        'replay': {
            'replay_class': 'UniformReplay',
            'batch_size': 512,
            'memory_size': int(1000000/3), # The total replay size is memory_size * replay_shards
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
        # If true, DDPG will expect an image at obs['pixel']['camera0']
        'pixel_input': False,
        'use_grayscale': False,
        # Stacks previous image frames together to provide history information
        'frame_stacks': 3,
        # Each action will be played this number of times. The reward of the consecutive actions will be the the reward
        # of the last action in the sequence
        'action_repeat': 1,
        # If false, the agent will send an image will be a list of frames to the replay.  When the learner receives an
        # observation, it will concatenate the frames into a single tensor.  This allows the replay to optimize memory
        # usage so that identical frames aren't duplicated in memory
        'frame_stack_concatenate_on_agent': False,
        # Debug only: agent will sleep for this number of seconds between actions
        'sleep_time': 0.0,
        # If an episode reaches this number of steps, the state will be considered terminal
        'limit_episode_length': 0, # 0 means no limit
        'video': {
            'record_video': False,
            'save_folder': None,
            'max_videos': 500,
            'record_every': 20,
        },
        # observation: if using FilterWrapper, any input not listed will be thrown out.
        # For example, if an observation had a value at obs['pixel']['badkey'], that value will be ignored
        'observation': {
            'pixel':['camera0', 'depth'],
            # if using ObservationConcatWrapper, all low_dim inputs will be concatenated together into a single input
            # named 'flat_inputs'
            'low_dim':['position', 'velocity', 'proprio', 'cube_pos', 'cube_quat', 'gripper_to_cube', 'low-dim'],
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
