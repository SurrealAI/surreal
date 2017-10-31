"""
Evaluates the test time performance of an agent.
"""
DEFAULT_DQN_CONFIG = {
    'lr': 5e-4,
    'max_timesteps': 100000,
    'buffer_size': 50000,
    'train_freq': 1,
    'batch_size': 32,
    'optimizer': 'Adam',
    'grad_norm_clipping': 10,
    'gamma': .99,
    'target_network_update_freq': 500,
    'double_q': True,
    'num_episodes_mean': 100, # average over last n episodes and report reward
    'checkpoint': {
        'dir': 'REQUIRED',
        'freq': 10000,
    },
    'log': {
        'freq': 1,
        'file_name': None,
        'file_mode': 'w',
        'time_format': None,
        'print_level': 'INFO',
        'stream': 'out',
    },
    'prioritized': {
        'enabled': True,
        'alpha': 0.6,
        'beta0': 0.4,
        'beta_anneal_iters': None,
        'eps': 1e-6
    },
}


if (C.checkpoint.freq is not None
    and T > self.learning_start
    and num_episodes > 100
    and T % C.checkpoint.freq == 0):
    if mean_reward > saved_mean_reward:
        if C.log.freq is not None:
            self.log.info("Saving model due to mean reward increase: "
                          "{:.1f} -> {:.1f}",
                          saved_mean_reward, mean_reward)
        self.save()
        saved_mean_reward = mean_reward
self.log.info('Training completed successfully!')
self.log.info('last 10 episode reward\t{:.1f}', np.mean(info['rewards'][-10:]))
