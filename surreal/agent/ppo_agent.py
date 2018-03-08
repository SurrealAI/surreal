"""
Actor function
"""
import torch
from torch.autograd import Variable
from .base import Agent
from surreal.model.ppo_net import PPOModel, DiagGauss
import surreal.utils as U
import numpy as np
from surreal.session import ConfigError
import time

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
            use_cuda = False,
        )
        self.total_exp_gen = 0 
        self.pd = DiagGauss(self.action_dim)

    def act(self, obs):
        obs = U.to_float_tensor(obs)
        assert torch.is_tensor(obs)
        obs = Variable(obs.unsqueeze(0))
        action_pd = self.model.forward_actor(obs).data.numpy()

        if self.agent_mode != 'eval_deterministic':
            action_choice = self.pd.sample(action_pd)
        else:
            action_choice = self.pd.maxprob(action_pd)
        np.clip(action_choice, -1, 1, out=action_choice)
        
        action_choice = action_choice.reshape((-1,))
        action_pd     = action_pd.reshape((-1,))
        self.total_exp_gen += 1

        if self.agent_mode != 'training':
            return action_choice
        else: 
            time.sleep(self.env_config.sleep_time)
            return action_choice, action_pd

    def fetch_parameter(self): 
        """
            Extends base class fetch_parameter to implement agent side bottlenecking
        """
        params, info = self._ps_client.fetch_parameter_with_info()
        if  self.agent_mode == 'training' and self.session_config.sync.if_sync and \
            self.total_exp_gen >= self.session_config.sync.agent_roll_max and \
            self.total_exp_gen > 0:
                print('Agent hanging after {} total steps'.format(self.total_exp_gen))
                start_hang_time = time.time()
                while not params:
                    time.sleep(self.session_config.sync.ping_freq)
                    params, info = self._ps_client.fetch_parameter_with_info()
                print('Agent restarted!, hung for {}s'.format(time.time()-start_hang_time))
                print('-----------')

        if params:
            self.on_parameter_fetched(params, info)
            self.total_exp_gen = 0 
        '''
            if eval: regular pull 
            if and using sync -- loopy pull 
        '''

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
