"""
Actor function
"""
import torch
import random
from torch.autograd import Variable
import surreal.utils as U
from surreal.model.q_net import build_ffqfunc
from .base import Agent


class QAgent(Agent):
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 agent_id,
                 agent_mode,
                 render=False):
        super().__init__(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config,
            agent_id=agent_id,
            agent_mode=agent_mode,
            render=render,
        )
        self.q_func, self.action_dim = build_ffqfunc(
            self.learner_config,
            self.env_config
        )
        self.exploration = self.get_exploration_schedule()
        self.T = 0

    def act(self, obs):
        assert torch.is_tensor(obs)
        if self.agent_mode == 'training':
            eps = self.exploration.value(self.T)
        else:
            eps = self.learner_config.eval.eps
        self.T += 1
        if (self.agent_mode in ['eval_deterministic', 'eval_deterministic_local']) or random.random() > eps:
            obs = obs[None]  # vectorize
            obs = Variable(obs, volatile=True)
            q_values = self.q_func(obs)
            return U.to_scalar(q_values.data.max(1)[1].cpu())
        else:  # random exploration
            return random.randrange(self.action_dim)

    def module_dict(self):
        return {
            'q_func': self.q_func
        }

    def default_config(self):
        return {
            'model': {
                'convs': '_list_',
                'fc_hidden_sizes': '_list_',
                'dueling': '_bool_'
            },
            'eval': {
                'eps': '_float_'
            }
        }

    def get_exploration_schedule(self):
        C = self.learner_config.algo.exploration
        if C.schedule.lower() == 'linear':
            return U.LinearSchedule(
                initial_p=1.0,
                final_p=C.final_eps,
                schedule_timesteps=int(C.steps),
            )
        else:
            steps = C.steps
            final_epses = C.final_eps
            U.assert_type(steps, list)
            U.assert_type(final_epses, list)
            assert len(steps) == len(final_epses)
            endpoints = [(0, 1.0)]
            for step, eps in zip(steps, final_epses):
                endpoints.append((step, eps))
            return U.PiecewiseSchedule(
                endpoints=endpoints,
                outside_value=final_epses[-1]
            )

