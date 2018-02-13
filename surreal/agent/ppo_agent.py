"""
Actor function
"""
import torch
from torch.autograd import Variable
from .base import Agent, AgentMode
from surreal.model.ppo_net import PPOModel, DiagGauss
import numpy as np
from surreal.session import ConfigError


class PPOAgent(Agent):

    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 agent_id,
                 agent_mode):
        super().__init__(
            learner_config=learner_config,
            env_config=env_config,
            session_config=session_config,
            agent_id=agent_id,
            agent_mode=agent_mode,
        )
        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_dim = self.env_config.obs_spec.dim[0]
        self.use_z_filter = self.learner_config.algo.use_z_filter
        self.init_log_sig = self.learner_config.algo.init_log_sig

        self.model = PPOModel(
            init_log_sig=self.init_log_sig,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            use_z_filter=self.use_z_filter,
        )

        self.pd = DiagGauss(self.action_dim)

    def act(self, obs):
        assert torch.is_tensor(obs)
        obs = Variable(obs.unsqueeze(0))
        action_pd = self.model.actor(obs).data.numpy()

        if self.agent_mode is not AgentMode.eval_deterministic:
            action_choice = self.pd.sample(action_pd)
        else:
            action_choice = self.pd.maxprob(action_pd)
        np.clip(action_choice, -1, 1, out=action_choice)
        
        action_choice = action_choice.reshape((-1,))
        action_pd     = action_pd.reshape((-1,))
        return (action_choice, action_pd)

    def module_dict(self):
        return {
            'ppo': self.model,
        }

    def default_config(self):
        return {
            'model': {
                'convs': '_list_',
                'fc_hidden_sizes': '_list_',
            },
        }

    def reset(self):
        pass
