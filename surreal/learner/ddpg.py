import torch
import torch.nn as nn
import surreal.utils as U
from surreal.utils.pytorch import GpuVariable as Variable
from surreal.session import PeriodicTracker
from easydict import EasyDict
from .base import Learner


class DDPGLearner(Learner):

    def __init__(self, config, model):
        super().__init__(config, model)

        self.discount_factor = 0.99

        self.model = model  # nothing but an alias
        self.model_target = self.model.clone()

        self.critic_criterion = nn.MSELoss()

        self.critic_optim = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=1e-3,
            eps=1e-4
        )

        self.actor_optim = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=1e-4,
            eps=1e-4
        )

        self.target_update_tracker = PeriodicTracker(
            period=self.config.target_network_update_freq,
        )

    def _update_target(self):
        self.model_target.copy_from(self.model)

    def _run_optimizer(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        norm_clip = self.config.grad_norm_clipping
        if norm_clip is not None:
            self.model.clip_grad_norm(norm_clip)
        self.optimizer.step()

    def _optimize(self, obses_t, actions, rewards, obses_tp1, dones):

        # estimate rewards using the next state: r + argmax_a Q'(s_{t+1}, u'(a))
        obses_tp1.volatile = True
        next_actions_target = self.model_target.forward_actor(obses_tp1)

        obses_tp1.volatile = False
        next_Q_target = self.model_target.forward_critic(obses_tp1, next_actions_target)
        next_Q_target.volatile = False
        y = rewards + self.discount_factor * next_Q_target * dones

        # compute Q(s_t, a_t)
        actions = actions.squeeze()
        y_policy = self.model.forward_critic(obses_t, actions)

        # critic update
        self.model.critic.zero_grad()
        critic_loss = self.critic_criterion(y_policy, y)
        critic_loss.backward()
        self.critic_optim.step()

        # actor update
        self.model.actor.zero_grad()
        actor_loss = -self.model.forward_critic(
            obses_t,
            self.model.forward_actor(obses_t)
        ).mean()
        actor_loss.backward()
        self.actor_optim.step()

        print('update steps')

    def learn(self, batch_exp, batch_i):
        self._optimize(
            batch_exp.obses[0],
            batch_exp.actions,
            batch_exp.rewards,
            batch_exp.obses[1],
            batch_exp.dones
        )
        batch_size = batch_exp.obses[0].size(0)
        if self.target_update_tracker.track_increment(batch_size):
            # Update target network periodically.
            self._update_target()