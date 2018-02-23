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
    parser.add_argument('--gpu', type=int, default=-1, help='device id for the gpu to use, -1 for cpu')

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
            }
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
            'use_z_filter': True,
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
            'limit_training_episode_length': 0, # 0 means no limit
            'agent_sleep_time': 1/50.0,
            'n_step': 5,
            'experience': 'ExpSenderWrapperMultiStepMovingWindow',
            # 'experience': 'ExpSenderWrapperSSARNStepBoostrap',
            'stride': 1,
        },
        'replay': {
            'replay_class': 'UniformReplay',
            'batch_size': 512,
            'memory_size': 1000000,
            'sampling_start_size': 1000,
        },
        'eval': {
            'eps': 0.05  # 5% random action under eval_stochastic mode
        }
    }

    env_config = {
        'env_name': args.env,
        'video': {
            'record_video': False,
            'save_directory': '/mnt/snaps/',
            'max_videos': 100,
            'record_every': 100,
        }
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
            'flush_iteration': 100,
        },
        'learner': {
            'gpu': args.gpu,
        },
    })

    session_config.extend(LOCAL_SESSION_CONFIG)
    return learner_config, env_config, session_config
