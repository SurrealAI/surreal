from surreal.session import (
    Config,
    LOCAL_SESSION_CONFIG,
    BASE_LEARNER_CONFIG,
    BASE_ENV_CONFIG
    )
from surreal.agent import PPOAgent
from surreal.learner import PPOLearner
from surreal.replay import FIFOReplay
from surreal.launch import SurrealDefaultLauncher
from surreal.env import make_env, make_env_config
import argparse


PPO_DEFAULT_LEARNER_CONFIG = Config({
    'model': {
        'convs': [],  # this can wait until TorchX
        'actor_fc_hidden_sizes': [300, 200],
        'critic_fc_hidden_sizes': [300, 200],
        'cnn_feature_dim': 256,
        'use_layernorm': False,
    },
    'algo': {
        # base configs
        # 'agent_class': 'PPOAgent',
        # 'learner_class': 'PPOLearner',
        # 'experience': 'ExpSenderWrapperMultiStepMovingWindowWithInfo',
        'use_z_filter': False,
        'use_r_filter': False,
        'gamma': .995,
        'n_step': 20,  # 10 for without RNN
        'stride': 15,  # 10 for without RNN
        'network': {
            'lr_actor': 0.75e-4,
            'lr_critic': 0.75e-4,
            'clip_actor_gradient': True,
            'actor_gradient_norm_clip': 3.,
            'clip_critic_gradient': True,
            'critic_gradient_norm_clip': 3.,
            'actor_regularization': 0.0,
            'critic_regularization': 0.0,
            'anneal': {
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
            'reward_scale': 0.005,
        },
        'rnn': {
            'if_rnn_policy': True,
            'rnn_hidden': 100,
            'rnn_layer': 1,
            'horizon': 5,
        },
        'consts': {
            'init_log_sig': -1.5,
            'log_sig_range': 0.25,
            'epoch_policy': 10,
            'epoch_baseline': 10,
            'adjust_threshold': (0.5, 2.0),  # threshold to magnify clip epsilon
            'kl_target': 0.01,  # target KL divergence between before and after
        },
        'adapt_consts': {
            'kl_cutoff_coeff': 250,  # penalty coeff when kl large
            'beta_init': 1.0,  # original beta
            'beta_range': (1/35.0, 35.0),  # range of the adapted penalty factor
            'scale_constant': 1.5,
        },
        'clip_consts': {
            'clip_epsilon_init': 0.2,  # factor of clipped loss
            'clip_range': (0.05, 0.3),  # range of the adapted penalty factor
            'scale_constant': 1.2,
        },

    },
    'replay': {
        # 'replay_class': 'FIFOReplay',
        'batch_size': 64,
        'memory_size': 96,
        'sampling_start_size': 64,
        'replay_shards': 1,
    },
    'parameter_publish': {
        'exp_interval': 4096,
    },
})
PPO_DEFAULT_LEARNER_CONFIG.extend(BASE_LEARNER_CONFIG)

PPO_DEFAULT_ENV_CONFIG = Config({
    'env_name': '',
    'action_repeat': 10,
    'pixel_input': False,
    'use_grayscale': False,
    'use_depth': False,
    'frame_stacks': 1,
    'sleep_time': 0,
    'video': {
        'record_video': False,
        'save_folder': None,
        'max_videos': 500,
        'record_every': 5,
    },
    'observation': {
        'pixel': ['camera0'],
        'low_dim':['robot-state','object-state'],
    },
    'eval_mode': {
        'demonstration': None
    },
    'demonstration': {
        'use_demo': False,
        'adaptive': True,
        # params for open loop reverse curriculum
        'increment_frequency': 100,
        'sample_window_width': 25,
        'increment': 25,

        # params for adaptive curriculum
        'mixing': ['random'],
        'mixing_ratio': [1.0],
        'ratio_step': [0.0],
        'improve_threshold': 0.1,
        'curriculum_length': 50,
        'history_length': 20,
    },
    'limit_episode_length': 0,
    'stochastic_eval': True,
})
PPO_DEFAULT_ENV_CONFIG.extend(BASE_ENV_CONFIG)

PPO_DEFAULT_SESSION_CONFIG = Config({
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
    'agent': {
        'fetch_parameter_mode': 'step',
        'fetch_parameter_interval': 100,  # 10 for without RNN
        'num_gpus': 0,
    },
    'sender': {
        'flush_iteration': 3,
    },
    'learner': {
        'num_gpus': 0,
    },
    'replay': {
        'max_puller_queue': 3,
        'max_prefetch_queue': 1,
    },
    'checkpoint': {
        'learner': {
            'mode': 'history',
            'periodic': 1000, # Save every 1000 steps
            'min_interval': 15 * 60, # No checkpoint less than 15 min apart.
        },
    },
})
PPO_DEFAULT_SESSION_CONFIG.extend(LOCAL_SESSION_CONFIG)


class PPOLauncher(SurrealDefaultLauncher):
    def __init__(self):
        learner_class = PPOLearner
        agent_class = PPOAgent
        replay_class = FIFOReplay
        learner_config = PPO_DEFAULT_LEARNER_CONFIG
        env_config = PPO_DEFAULT_ENV_CONFIG
        session_config = PPO_DEFAULT_SESSION_CONFIG
        super().__init__(agent_class,
                         learner_class,
                         replay_class,
                         session_config,
                         env_config,
                         learner_config)

    def setup(self, argv):
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
                            help='session_config.folder that has experiment'
                            'files like checkpoint and logs')
        parser.add_argument('--agent-batch', type=int, default=1,
                            help='how many agents/evals per batch')
        parser.add_argument('--unit-test', action='store_true',
                            help='Set config values to settings that can run locally for unit testing')

        args = parser.parse_args(args=argv)

        self.env_config.env_name = args.env
        self.env_config = make_env_config(self.env_config)

        self.session_config.folder = args.experiment_folder
        self.session_config.agent.num_gpus = args.agent_num_gpus
        self.session_config.learner.num_gpus = args.num_gpus
        if args.restore_folder is not None:
            self.session_config.checkpoint.restore = True
            self.session_config.checkpoint.restore_folder = args.restore_folder
        self.agent_batch_size = args.agent_batch
        self.eval_batch_size = args.agent_batch

        if args.unit_test:
            self.learner_config.replay.batch_size = 2
            self.learner_config.replay.sampling_start_size = 2


def main():
    PPOLauncher().main()


if __name__ == '__main__':
    main()


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
