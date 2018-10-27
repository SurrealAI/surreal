"""
Actor function
"""
import time
import torch
import numpy as np
import torchx as tx
import torchx.nn as nnx
import surreal.utils as U
from surreal.model.ppo_net import PPOModel, DiagGauss
from surreal.env import ExpSenderWrapperMultiStepMovingWindowWithInfo
from surreal.session import ConfigError
from .base import Agent


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
        self.action_dim = self.env_config.action_spec.dim[0]
        self.obs_spec = self.env_config.obs_spec
        self.use_z_filter = self.learner_config.algo.use_z_filter

        self.init_log_sig = self.learner_config.algo.consts.init_log_sig
        self.log_sig_range = self.learner_config.algo.consts.log_sig_range

        # setting agent mode
        if self.agent_mode != 'training':
            if self.env_config.stochastic_eval:
                self.agent_mode = 'eval_stochastic'
            else:
                self.agent_mode = 'eval_deterministic'

        if self.agent_mode != 'training':
            self.noise = 0
        else:
            self.noise = np.random.uniform(low=-self.log_sig_range,
                                           high=self.log_sig_range)
        self.rnn_config = self.learner_config.algo.rnn

        # GPU setup
        # TODO: deprecate
        self._num_gpus = session_config.agent.num_gpus

        if torch.cuda.is_available():
            self.gpu_ids = 'cuda:all'
            self.log.info('PPO agent is using GPU')
            # Note that user is responsible for only providing one GPU for the program
            self.log.info('cudnn version: {}'.format(torch.backends.cudnn.version()))
            torch.backends.cudnn.benchmark = True
        else:
            self.gpu_ids = 'cpu'
            self.log.info('PPO agent is using CPU')

        self.pd = DiagGauss(self.action_dim)
        self.cells = None

        with tx.device_scope(self.gpu_ids):
            if self.rnn_config.if_rnn_policy:
                # Note that .detach() is necessary here to prevent overflow of memory
                # otherwise rollout in length of thousands will prevent previously
                # accumulated hidden/cell states from being freed.
                self.cells = (torch.zeros(self.rnn_config.rnn_layer,
                                          1,  # batch_size is 1
                                          self.rnn_config.rnn_hidden).detach(),
                              torch.zeros(self.rnn_config.rnn_layer,
                                          1,  # batch_size is 1
                                          self.rnn_config.rnn_hidden).detach())

            self.model = PPOModel(
                obs_spec=self.obs_spec,
                action_dim=self.action_dim,
                model_config=self.learner_config.model,
                use_cuda=False,
                init_log_sig=self.init_log_sig,
                use_z_filter=self.use_z_filter,
                if_pixel_input=self.env_config.pixel_input,
                rnn_config=self.rnn_config,
            )

    def act(self, obs):
        '''
            Agent returns an action based on input observation. if in training,
            returns action along with action infos, which includes the current
            probability distribution, RNN hidden states and etc.
            Args:
                obs: numpy array of (1, obs_dim)

            Returns:
                action_choice: sampled or max likelihood action to input to env
                action_info: list of auxiliary information - [onetime, persistent]
                    Note: this includes probability distribution the action is
                    sampled from, RNN hidden states
        '''
        # Note: we collect two kinds of action infos, one persistent one onetime
        # persistent info is collected for every step in rollout (i.e. policy probability distribution)
        # onetime info is collected for the first step in partial trajectory (i.e. RNN hidden state)
        # see ExpSenderWrapperMultiStepMovingWindowWithInfo in exp_sender_wrapper for more
        action_info = [[], []]

        with tx.device_scope(self.gpu_ids):
            obs_tensor = {}
            for mod in obs.keys():
                obs_tensor[mod] = {}
                for k in obs[mod].keys():
                    obs_tensor[mod][k] = torch.tensor(obs[mod][k], dtype=torch.float32).unsqueeze(0)

            if self.rnn_config.if_rnn_policy:
                action_info[0].append(self.cells[0].squeeze(1).cpu().numpy())
                action_info[0].append(self.cells[1].squeeze(1).cpu().numpy())

            action_pd, self.cells = self.model.forward_actor_expose_cells(obs_tensor, self.cells)
            action_pd = action_pd.detach().cpu().numpy()
            action_pd[:, self.action_dim:] *= np.exp(self.noise)

            if self.agent_mode != 'eval_deterministic':
                action_choice = self.pd.sample(action_pd)
            else:
                action_choice = self.pd.maxprob(action_pd)
            np.clip(action_choice, -1, 1, out=action_choice)

            action_choice = action_choice.reshape((-1,))
            action_pd     = action_pd.reshape((-1,))
            action_info[1].append(action_pd)
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
            reset of LSTM hidden and cell states
        '''
        if self.rnn_config.if_rnn_policy:
            # Note that .detach() is necessary here to prevent overflow of memory
            # otherwise rollout in length of thousands will prevent previously
            # accumulated hidden/cell states from being freed.
            with tx.device_scope(self.gpu_ids):
                self.cells = (torch.zeros(self.rnn_config.rnn_layer,
                                          1,  # batch_size is 1
                                          self.rnn_config.rnn_hidden).detach(),
                              torch.zeros(self.rnn_config.rnn_layer,
                                          1,  # batch_size is 1
                                          self.rnn_config.rnn_hidden).detach())

    def prepare_env_agent(self, env):
        env = super().prepare_env_agent(env)
        env = ExpSenderWrapperMultiStepMovingWindowWithInfo(env,
                                                            self.learner_config,
                                                            self.session_config)
        return env
