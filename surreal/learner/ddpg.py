import torch
import torch.nn as nn
import numpy as np
import surreal.utils as U
from surreal.model.ddpg_net import DDPGModel
from .base import Learner
from .aggregator import NstepReturnAggregator, SSARConcatAggregator
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG, BASE_LEARNER_CONFIG

class DDPGLearner(Learner):

    def __init__(self, learner_config, env_config, session_config):
        super().__init__(learner_config, env_config, session_config)
        learner_config = Config(learner_config).extend(BASE_LEARNER_CONFIG)
        self.tau = 0.01
        self.discount_factor = learner_config.algo.gamma
        self.n_step = learner_config.algo.n_step

        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_dim = self.env_config.obs_spec.dim[0]

        self.model = DDPGModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )

        self.model_target = DDPGModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )

        self.critic_criterion = nn.MSELoss()

        self.critic_optim = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=1e-3
        )

        self.actor_optim = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=1e-4
        )

        # self.aggregator = NstepReturnAggregator(self.env_config.obs_spec, self.env_config.action_spec, self.discount_factor)
        self.aggregator = SSARConcatAggregator(self.env_config.obs_spec, self.env_config.action_spec)

        U.hard_update(self.model_target.actor, self.model.actor)
        U.hard_update(self.model_target.critic, self.model.critic)
        # self.train_iteration = 0

    def _optimize(self, obs, actions, rewards, obs_next, done):

        assert actions.max().data[0] <= 1.0
        assert actions.min().data[0] >= -1.0

        # estimate rewards using the next state: r + argmax_a Q'(s_{t+1}, u'(a))
        # obs_next.volatile = True
        next_actions_target = self.model_target.forward_actor(obs_next)

        # obs_next.volatile = False
        next_Q_target = self.model_target.forward_critic(obs_next, next_actions_target)
        # next_Q_target.volatile = False
        y = rewards + pow(self.discount_factor, self.n_step) * next_Q_target * (1.0 - done)
        y = y.detach()

        # print('next_Q_target', next_Q_target)
        # print('y', y)

        # compute Q(s_t, a_t)
        y_policy = self.model.forward_critic(
            obs.detach(),
            actions.detach()
        )

        # critic update
        self.model.critic.zero_grad()
        critic_loss = self.critic_criterion(y_policy, y)
        critic_loss.backward()
        # for p in self.model.critic.parameters():
        #     p.grad.data.clamp_(-5.0, 5.0)
        self.critic_optim.step()

        # actor update
        self.model.actor.zero_grad()
        actor_loss = -self.model.forward_critic(
            obs.detach(),
            self.model.forward_actor(obs.detach())
        ).mean()
        actor_loss.backward()
        # for p in self.model.actor.parameters():
        #     p.grad.data.clamp_(-5.0, 5.0)
        self.actor_optim.step()

        self.update_tensorplex({
            'actor_loss': actor_loss.data[0],
            'critic_loss': critic_loss.data[0],
            'action_norm': actions.norm(2, 1).mean().data[0],
            'rewards': rewards.mean().data[0],
            'Q_target': y.mean().data[0],
            'Q_policy': y_policy.mean().data[0],
        })
        # soft update target networks
        U.soft_update(self.model_target.actor, self.model.actor, self.tau)
        U.soft_update(self.model_target.critic, self.model.critic, self.tau)

    def learn(self, batch):
        # for i in range(len(batch)):
        #     exp = batch[i]
        #     k = 0
        #     batch[i] = {
        #         'obs': [exp['obs_arr'][k], exp['obs_arr'][k + 1]],
        #         'action': exp['action_arr'][0],
        #         'reward': exp['reward_arr'][0],
        #         'done': exp['done_arr'][0],
        #         'info': exp['info_arr'][0]
        #     }
        batch = self.aggregator.aggregate(batch)
        # for k in batch:
        #     print(k, type(batch[k]), batch[k].shape)
        self._optimize(
            batch.obs,
            batch.actions,
            batch.rewards,
            batch.obs_next,
            batch.dones
        )

    def module_dict(self):
        return {
            'ddpg': self.model,
        }