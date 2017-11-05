import torch
import surreal.utils as U
from surreal.utils.pytorch import GpuVariable as Variable
from surreal.session import PeriodicTracker
from easydict import EasyDict
from .base import Learner


class DQN(Learner):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.q_func = model  # nothing but an alias
        self.q_target = self.q_func.clone()
        self.optimizer = torch.optim.Adam(
            self.q_func.parameters(),
            lr=self.config.lr,
            eps=1e-4
        )
        self.target_update_tracker = PeriodicTracker(
            period=self.config.target_network_update_freq,
        )

    def _update_target(self):
        self.q_target.copy_from(self.q_func)

    def _run_optimizer(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        norm_clip = self.config.grad_norm_clipping
        if norm_clip is not None:
            self.q_func.clip_grad_norm(norm_clip)
            # torch.nn.utils.net_clip_grad_norm(
            #     self.q_func.parameters(),
            #     max_norm=norm_clip
            # )
        self.optimizer.step()

    def _optimize(self, obses_t, actions, rewards, obses_tp1, dones, weights):
        # Compute Q(s_t, a)
        # columns of actions taken
        C = self.config
        batch_size = obses_t.size(0)
        assert (U.shape(actions)
                == U.shape(rewards)
                == U.shape(dones)
                == (batch_size, 1))
        q_t_at_action = self.q_func(obses_t).gather(1, actions)
        q_tp1 = self.q_target(obses_tp1)
        # Double Q
        if C.double_q:
            # select argmax action using online weights instead of q_target
            q_tp1_online = self.q_func(obses_tp1)
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
        q_expected = rewards + C.gamma * q_tp1_best
        td_error = q_t_at_action - q_expected
        # torch_where
        raw_loss = U.huber_loss_per_element(td_error)
        weighted_loss = torch.mean(weights * raw_loss)
        self._run_optimizer(weighted_loss)
        return td_error

    def learn(self, batch_exp, batch_i):
        weights = Variable(U.torch_ones_like(batch_exp.rewards))
        td_errors = self._optimize(
            batch_exp.obses[0],
            batch_exp.actions,
            batch_exp.rewards,
            batch_exp.obses[1],
            batch_exp.dones,
            weights,
        )
        batch_size = batch_exp.obses[0].size(0)
        if self.target_update_tracker.track_increment(batch_size):
            # Update target network periodically.
            self._update_target()
        return td_errors

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
            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
        else:
            weights = Variable(U.torch_ones_like(exp.rewards))
            batch_idxes = None

        td_errors = self.optimize(
            exp.obses[0],
            exp.actions,
            exp.rewards,
            exp.obses[1],
            exp.dones,
            weights,
        )
        batch_size = exp.obses[0].size(0)
        if C.prioritized.enabled:
            # TODO
            new_priorities = torch.abs(td_errors) + C.prioritized.eps
            new_priorities = U.to_numpy(new_priorities)
            replay_buffer.update_priorities(batch_idxes, new_priorities)

        if self.target_update_tracker.track_increment(batch_size):
            # Update target network periodically.
            self._update_target()

    """
