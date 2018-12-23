import argparse
from surreal.session import (
    Config,
    LOCAL_SESSION_CONFIG,
    BASE_LEARNER_CONFIG,
    BASE_ENV_CONFIG
    )
from surreal.agent import DDPGAgent
from surreal.learner import DDPGLearner
from surreal.replay import UniformReplay
from surreal.launch import SurrealDefaultLauncher
from surreal.env import make_env_config

# TODO：Documentation on config files

DDPG_DEFAULT_LEARNER_CONFIG = Config({
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
            'max_sigma': 1.0,

            # Or, use Ornstein-Uhlenbeck noise instead of gaussian
            #'noise_type': 'ou_noise',
            'theta': 0.15,
            'dt': 1e-3,
        },
    },
    'replay': {
        'batch_size': 512,
        'memory_size': int(1000000/3), # The total replay size is memory_size * replay_shards
        'sampling_start_size': 3000,
        'replay_shards': 3,
    },
    'parameter_publish': {
        # Minimum amount of time (seconds) between two parameter publish
        'min_publish_interval': 3,
    },
})

DDPG_DEFAULT_LEARNER_CONFIG.extend(BASE_LEARNER_CONFIG)

DDPG_DEFAULT_ENV_CONFIG = Config({
    'env_name': '_str_',
    'num_agents': '_int_',

    'demonstration': None,
    'use_depth': False,
    'render': False,

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
    'frame_stack_concatenate_on_env': False,
    # Debug only: agent will sleep for this number of seconds between actions
    'sleep_time': 0.0,
    # If an episode reaches this number of steps, the state will be considered terminal
    'limit_episode_length': 0, # 0 means no limit
    'video': {
        'record_video': True,
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
        'low_dim':['position', 'velocity', 'proprio', 'robot-state', 'cube_pos', 'cube_quat', 'gripper_to_cube', 'low-dim'],
    },
})

DDPG_DEFAULT_ENV_CONFIG.extend(BASE_ENV_CONFIG)

DDPG_DEFAULT_SESSION_CONFIG = Config({
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
        'num_gpus': 0,
    },
    'sender': {
        'flush_iteration': 100,
    },
    'learner': {
        'prefetch_processes': 3,
        'num_gpus': 0,
    },
})

DDPG_DEFAULT_SESSION_CONFIG.extend(LOCAL_SESSION_CONFIG)

DDPG_BLOCK_LIFTING_LEARNER_CONFIG = Config({
    'algo': {
        'network': {
            'lr_actor': 1e-4,
            'lr_critic': 1e-4,
            # Weight regularization
            'actor_regularization': 1e-4,
            'critic_regularization': 1e-4,
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
            'max_sigma': 2.0,
            # Use Ornstein-Uhlenbeck noise instead of gaussian
            'noise_type': 'ou_noise',
            'theta': 0.15,
            'dt': 1e-3,
        },
    },
})

DDPG_BLOCK_LIFTING_LEARNER_CONFIG.extend(DDPG_DEFAULT_LEARNER_CONFIG)

DDPG_BLOCK_LIFTING_ENV_CONFIG = Config({
    'env_name': '_str_',
    'num_agents': '_int_',
    # If true, DDPG will expect an image at obs['pixel']['camera0']
    'pixel_input': True,
    'use_grayscale': False,
    # Stacks previous image frames together to provide history information
    'frame_stacks': 3,
    # Each action will be played this number of times. The reward of the consecutive actions will be the the reward
    # of the last action in the sequence
    'action_repeat': 10,
    # If an episode reaches this number of steps, the state will be considered terminal
    'limit_episode_length': 200, # 0 means no limit
    # observation: if using FilterWrapper, any input not listed will be thrown out.
    # For example, if an observation had a value at obs['pixel']['badkey'], that value will be ignored
    'observation': {
        'pixel':['camera0', 'depth'],
        # if using ObservationConcatWrapper, all low_dim inputs will be concatenated together into a single input
        # named 'flat_inputs'
        'low_dim':['position', 'velocity', 'robot-state', 'proprio', 'cube_pos', 'cube_quat', 'gripper_to_cube', 'low-dim'],
    },
})

DDPG_BLOCK_LIFTING_ENV_CONFIG.extend(DDPG_DEFAULT_ENV_CONFIG)

class DDPGLauncher(SurrealDefaultLauncher):
    def __init__(self):
        learner_class = DDPGLearner
        agent_class = DDPGAgent
        replay_class = UniformReplay
        learner_config = DDPG_DEFAULT_LEARNER_CONFIG
        env_config = DDPG_DEFAULT_ENV_CONFIG
        session_config = DDPG_DEFAULT_SESSION_CONFIG
        super().__init__(agent_class,
                         learner_class,
                         replay_class,
                         session_config,
                         env_config,
                         learner_config)

    def setup(self, argv):
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
        parser.add_argument('--restore-folder', type=str, default=None,
                            help='folder containing checkpoint to restore from')
        parser.add_argument('--experiment-folder', required=True,
                            help='session_config.folder that has experiment files'
                            ' like checkpoint and logs')
        parser.add_argument('--agent-batch', type=int, default=1,
                            help='how many agents/evals per batch')
        parser.add_argument('--eval-batch', type=int, default=1,
                            help='how many agents/evals per batch')
        parser.add_argument('--unit-test', action='store_true',
                            help='Prevents sharding replay and paramter '
                            'server. Helps prevent address collision'
                            ' in unit testing.')

        args = parser.parse_args(args=argv)

        self.env_config.env_name = args.env
        self.env_config = make_env_config(self.env_config)
        self.env_config.num_agents = args.num_agents

        self.session_config.folder = args.experiment_folder
        self.session_config.agent.num_gpus = args.agent_num_gpus
        self.session_config.learner.num_gpus = args.num_gpus
        if args.restore_folder is not None:
            self.session_config.checkpoint.restore = True
            self.session_config.checkpoint.restore_folder = args.restore_folder
        self.agent_batch_size = args.agent_batch
        self.eval_batch_size = args.eval_batch

        # Used in tests: Prevent IP address in use error
        #                Prevent replay from hanging learner
        #                due to sample_start
        if args.unit_test:
            self.learner_config.replay.sampling_start_size = 5
            self.learner_config.replay.replay_shards = 1
            self.session_config.ps.shards = 1


def main():
    DDPGLauncher().main()

if __name__ == '__main__':
    main()

