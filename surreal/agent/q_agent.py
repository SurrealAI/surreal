"""
Actor function
"""
import torch
import random
from torch.autograd import Variable
import surreal.utils as U
from surreal.model.q_net import FFQfunc
from .base import Agent, AgentMode


class QAgent(Agent):
    def __init__(self, config, agent_mode):
        super().__init__(config, agent_mode)
        self.q_func = FFQfunc(**self.config.model)
        self.exploration = self.get_exploration_schedule()
        # TODO move this to env config
        self.action_dim = self.config.model.action_dim
        self.T = 0

    def act(self, obs):
        assert torch.is_tensor(obs)
        eps = self.exploration.value(self.T)
        self.T += 1
        if (self.agent_mode == AgentMode.eval_deterministic
            or random.random() > eps):
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
        
    def get_exploration_schedule(self):
        C = self.config.algo.exploration
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

