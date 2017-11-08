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

    def _optimize(self, obs, actions, rewards, obs_next, dones):

        # estimate rewards using the next state: r + argmax_a Q'(s_{t+1}, u'(a))
        obs_next.volatile = True
        next_actions_target = self.model_target.forward_actor(obs_next)

        obs_next.volatile = False
        next_Q_target = self.model_target.forward_critic(obs_next, next_actions_target)
        next_Q_target.volatile = False
        y = rewards + self.discount_factor * next_Q_target * dones

        # compute Q(s_t, a_t)
        actions = actions.squeeze()
        y_policy = self.model.forward_critic(obs, actions)

        # critic update
        self.model.critic.zero_grad()
        critic_loss = self.critic_criterion(y_policy, y)
        critic_loss.backward()
        self.critic_optim.step()

        # actor update
        self.model.actor.zero_grad()
        actor_loss = -self.model.forward_critic(
            obs,
            self.model.forward_actor(obs)
        ).mean()
        actor_loss.backward()
        self.actor_optim.step()

    def learn(self, batch_exp, batch_i):
        self._optimize(
            batch_exp.obs,
            batch_exp.actions,
            batch_exp.rewards,
            batch_exp.obs_next,
            batch_exp.dones
        )
        if self.target_update_tracker.track_increment(1):
            # Update target network periodically.
            self._update_target()