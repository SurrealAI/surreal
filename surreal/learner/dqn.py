import torch
import surreal.utils as U
#from surreal.utils.pytorch import GpuVariable as Variable
from surreal.session import PeriodicTracker
from surreal.model.q_net import build_ffqfunc
from .base import Learner


class DQNLearner(Learner):
    def __init__(self, learner_config, env_config, session_config):
        super().__init__(learner_config, env_config, session_config)
        self.q_func, self.action_dim = build_ffqfunc(
            self.learner_config,
            self.env_config
        )
        self.algo = self.learner_config.algo
        self.q_target = self.q_func.clone()
        self.optimizer = torch.optim.Adam(
            self.q_func.parameters(),
            lr=self.algo.lr,
            eps=1e-4
        )
        self.target_update_tracker = PeriodicTracker(
            period=self.algo.target_network_update_freq,
        )

    def _update_target(self):
        self.q_target.copy_from(self.q_func)

    def _run_optimizer(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        norm_clip = self.algo.grad_norm_clipping
        if norm_clip is not None:
            self.q_func.clip_grad_norm(norm_clip)
            # torch.nn.utils.net_clip_grad_norm(
            #     self.q_func.parameters(),
            #     max_norm=norm_clip
            # )
        self.optimizer.step()

    def _optimize(self, obs, actions, rewards, obs_next, dones, weights):
        # Compute Q(s_t, a)
        # columns of actions taken
        batch_size = obs.size(0)
        assert (U.shape(actions)
                == U.shape(rewards)
                == U.shape(dones)
                == (batch_size, 1))
        q_t_at_action = self.q_func(obs).gather(1, actions)
        q_tp1 = self.q_target(obs_next)
        # Double Q
        if self.algo.double_q:
            # select argmax action using online weights instead of q_target
            q_tp1_online = self.q_func(obs_next)
            q_tp1_online_argmax = q_tp1_online.max(1, keepdim=True)[1]
            q_tp1_best = q_tp1.gather(1, q_tp1_online_argmax)
        else:
            # Minh 2015 Nature paper
            # use target network for both policy and value selection
            q_tp1_best = q_tp1.max(1, keepdim=True)[0]
        # Q value for terminal states are 0
        q_tp1_best = (1.0 - dones) * q_tp1_best
        # .detach() stops gradient and makes the Variable forget its creator
        q_tp1_best = q_tp1_best.detach()
        # RHS of bellman equation
        q_expected = rewards + self.algo.gamma * q_tp1_best
        td_error = q_t_at_action - q_expected
        # torch_where
        raw_loss = U.huber_loss_per_element(td_error)
        weighted_loss = torch.mean(weights * raw_loss)
        self._run_optimizer(weighted_loss)
        return td_error

    def learn(self, batch_exp):
        weights = (U.torch_ones_like(batch_exp.rewards))
        td_errors = self._optimize(
            batch_exp.obs,
            batch_exp.actions,
            batch_exp.rewards,
            batch_exp.obs_next,
            batch_exp.dones,
            weights,
        )
        batch_size = batch_exp.obs.size(0)
        if self.target_update_tracker.track_increment(batch_size):
            # Update target network periodically.
            self._update_target()
        mean_td_error = U.to_scalar(torch.mean(torch.abs(td_errors)))
        self.tensorplex.add_scalars({
            'td_error': mean_td_error
        })

    def default_config(self):
        return {
            'model': {
                'convs': '_list_',
                'fc_hidden_sizes': '_list_',
                'dueling': '_bool_'
            },
            'algo': {
                'lr': 1e-3,
                'optimizer': 'Adam',
                'grad_norm_clipping': 10,
                'gamma': .99,
                'target_network_update_freq': '_int_',
                'double_q': True,
                'exploration': {
                    'schedule': 'linear',
                    'steps': '_int_',
                    'final_eps': 0.01,
                },
                'prioritized': {
                    'enabled': False,
                    'alpha': 0.6,
                    'beta0': 0.4,
                    'beta_anneal_iters': None,
                    'eps': 1e-6
                },
            },
        }

    def module_dict(self):
        return {
            'q_func': self.q_func
        }

    """
    def train_batch(self, batch_i, exp):
        C = self.config

        if C.prioritized.enabled:
            # TODO
            replay_buffer = PrioritizedReplayBuffer(C.buffer_size,
                                                alpha=C.prioritized.alpha,
                                                obs_encode_mode=C.obs_encode_mode)
            beta_iters = C.prioritized.beta_anneal_iters
            if beta_iters is None:
                beta_iters = C.max_timesteps
            beta_schedule = U.LinearSchedule(schedule_timesteps=beta_iters,
                                             initial_p=C.prioritized.beta0,
                                             final_p=1.0)
        else:
            beta_schedule = None

        # TODO train_freq removed, is it useful at all?
        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        if C.prioritized.enabled:
            experience = self.replay.sample(C.batch_size,
                                              beta=beta_schedule.value(T))
            (obs, actions, rewards, obs_next, dones, weights, batch_idxes) = experience
        else:
            weights = Variable(U.torch_ones_like(exp.rewards))
            batch_idxes = None

        td_errors = self.optimize(
            exp.obs[0],
            exp.actions,
            exp.rewards,
            exp.obs[1],
            exp.dones,
            weights,
        )
        batch_size = exp.obs[0].size(0)
        if C.prioritized.enabled:
            # TODO
            new_priorities = torch.abs(td_errors) + C.prioritized.eps
            new_priorities = U.to_numpy(new_priorities)
            replay_buffer.update_priorities(batch_idxes, new_priorities)

        if self.target_update_tracker.track_increment(batch_size):
            # Update target network periodically.
            self._update_target()

    """
