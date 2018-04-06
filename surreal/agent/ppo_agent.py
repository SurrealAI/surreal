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
import resource

class PPOAgent(Agent):
    '''
        Class that specifies PPO agent logic
        Important attributes:
            init_log_sig: initial log sigma for diagonal gausian policy
            model: PPO_Model instance. see surreal.model.ppo_net
            pd: DiagGauss instance. see surreal.model.ppo_net
        Member functions:
            act
            reset
    '''
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
        self.init_log_sig = self.learner_config.algo.consts.init_log_sig
        self.rnn_config = self.learner_config.algo.rnn

        self.cells = None
        if self.rnn_config.if_rnn_policy:
            self.cells = (Variable(torch.zeros(self.rnn_config.rnn_layer, 
                                               1, # batch_size is 1
                                               self.rnn_config.rnn_hidden)),
                         Variable(torch.zeros(self.rnn_config.rnn_layer, 
                                              1, # batch_size is 1
                                              self.rnn_config.rnn_hidden)))

        self.model = PPOModel(
            init_log_sig=self.init_log_sig,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            use_z_filter=self.use_z_filter,
            rnn_config=self.rnn_config,
            use_cuda=False,
        )

        self.pd = DiagGauss(self.action_dim)

    def act(self, obs):
        '''
            Agent returns an action based on input observation. if in training,
            returns action along with action infos, which includes the current
            probability distribution, RNN hidden states and etc.
            Args:
                obs: numpy array of (1, obs_dim)

            Returns:
                action_choice: sampled or max likelihood action to input to env
                action_info: list of auxiliary information
                    Note: this includes probability distribution the action is
                    sampled from, RNN hidden states
        '''
        print('\tCheckpoint 1: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        obs = U.to_float_tensor(obs)
        assert torch.is_tensor(obs)
        obs = Variable(obs.unsqueeze(0))
        action_info = [[], []] # 0: one time, 1: every time
        print('\tCheckpoint 1.1: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        action_pd, self.cells = self.model.forward_actor_expose_cells(obs, self.cells)
        print('\tCheckpoint 1.2: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        action_pd = action_pd.data.numpy()
        print('\tCheckpoint 2: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        if self.rnn_config.if_rnn_policy:
            action_info[0].append(self.cells[0].squeeze(1).data.numpy())
            action_info[0].append(self.cells[1].squeeze(1).data.numpy())

        if self.agent_mode != 'eval_deterministic':
            action_choice = self.pd.sample(action_pd)
        else:
            action_choice = self.pd.maxprob(action_pd)
        np.clip(action_choice, -1, 1, out=action_choice)
        print('\tCheckpoint 3: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        
        action_choice = action_choice.reshape((-1,))
        action_pd     = action_pd.reshape((-1,))
        action_info[1].append(action_pd)
        print('\tCheckpoint 4: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if self.agent_mode != 'training':
            return action_choice
        else: 
            time.sleep(self.env_config.sleep_time)
            return action_choice, action_info

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
        '''
            Currently unimplemented. in the future it will contain reset of 
            RNN hidden states
        '''
        if self.rnn_config.if_rnn_policy:
            self.cells = (Variable(torch.zeros(self.rnn_config.rnn_layer, 
                                               1, # batch_size is 1
                                               self.rnn_config.rnn_hidden)),
                          Variable(torch.zeros(self.rnn_config.rnn_layer, 
                                               1, # batch_size is 1
                                               self.rnn_config.rnn_hidden)))

