"""
Aggregate experience tuple into pytorch-ready tensors
"""
import numpy as np
import copy
import collections
import torch
import surreal.utils as U
from surreal.env import ActionType

class FrameStackPreprocessor():
    """
        Accepts experience sent by SSAR experience senders
    """

    def __init__(self, frame_stacks):
        self.frame_stacks = frame_stacks
        self.printed = False

    def preprocess_obs(self, obs):
        # We must copy here because the experience sender should send the non-stacked version
        if 'pixel' in obs:
            for key in obs['pixel']:
                obs['pixel'][key] = np.concatenate(obs['pixel'][key], axis=0)
                assert len(obs['pixel'][key].shape) == 3

    def preprocess_list(self, exp_list):
        for exp in exp_list:  # dict
            for obs in (exp['obs'][0], exp['obs'][1]):
                self.preprocess_obs(obs)
        return exp_list

class SSARAggregator():
    """
        Accepts experience sent by SSAR experience senders
        aggregate() returns float arrays:
        {
            obs = batch_size * observation
            obs_next = batch_size * next_observation
            actions = batch_size * actions,
            rewards = batch_size * 1,
            dones = batch_size * 1,
        }
    """
    def __init__(self, obs_spec, action_spec):
        U.assert_type(obs_spec, dict)
        U.assert_type(action_spec, dict)
        self.action_type = ActionType[action_spec['type']]
        self.action_spec = action_spec
        self.obs_spec = obs_spec

    def aggregate(self, exp_list):
        # TODO add support for more diverse obs_spec and action_spec
        """

        Args:
            exp_list:
        
        Returns:
            aggregated experience
        """

        obs0, actions, rewards, obs1, dones = (
            collections.OrderedDict(), [], [], collections.OrderedDict(), [])
        i = 0
        for exp in exp_list:  # dict
            i += 1
            for modality in exp['obs'][0]:
                if modality not in obs0:
                    obs0[modality] = collections.OrderedDict()
                for key in exp['obs'][0][modality]:
                    if key not in obs0[modality]:
                        obs0[modality][key] = []
                    obs0[modality][key].append(np.array(exp['obs'][0][modality][key], copy=False))
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            for modality in exp['obs'][1]:
                if modality not in obs1:
                    obs1[modality] = collections.OrderedDict()
                for key in exp['obs'][1][modality]:
                    if key not in obs1[modality]:
                        obs1[modality][key] = []
                    obs1[modality][key].append(np.array(exp['obs'][1][modality][key], copy=False))
            dones.append(float(exp['done']))
        if self.action_type == ActionType.continuous:
            actions = np.array(actions, dtype=np.float32)
        elif self.action_type == ActionType.discrete:
            actions = np.array(actions, dtype=np.int32)
        else:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))

        for obs in obs0, obs1:
            for modality in obs:
                for key in obs[modality]:
                    obs[modality][key] = np.array(obs[modality][key])

        return {
            'obs': obs0,
            'obs_next': obs1,
            'actions': np.array(actions),
            'rewards': np.expand_dims(rewards, axis=1),
            'dones': np.expand_dims(dones, axis=1),
        }


class MultistepAggregatorWithInfo():
    """
        Accepts input by ExpSenderWrapperMultiStepMovingWindowWithInfo
        aggregate() returns float Tensors
        {
            obs = batch_size * n_step * observation
            next_obs = batch_size * 1 * next_observation
            actions = batch_size * n_step * actions,
            rewards = batch_size * n_step,
            persistent_infos = list of batched FloatTensor action info per step,
            onetime_infos = FloatTensor of action info for first step of batch,
            dones = batch_size * n_step,
        }

        This class of aggregator is used in companion with exp sender.
        ExpSenderWrapperMultiStepMovingWindowWithInfo. The key return values are
        a list of persistent infos, and a FloatTensor of onetime info that are 
        auxiliary information returned from the agent. The main difference is
        that persistent infos are required for every step, for instance the probability
        distribution for policy, whereas onetime infos are needed only for first or 
        last of the steps, for example the LSTM hidden/cell states.

        This aggregator should be used when the agent is required to 
        expose attributes to the learner. For instance, PPO needs to use this
        aggregator to send over the action probability distribution and optional
        RNN hidden states.

        This attribute will be returned as a list of batched FloatTensors. In
        the case of PPO without RNN policy, the action_infos attribute is of
        form:
            persistent info: [FloatTensor of shape (batch_size, 2 * act_dim)]
            onetime infos: []

        In case with RNN:
            persistent info: [FloatTensor of shape (batch, horizon, 2 * act_dim)]
            onetime infos: [LSTM hidden state, LSTM cell state]

    """
    def __init__(self, obs_spec, action_spec):
        U.assert_type(obs_spec, dict)
        U.assert_type(action_spec, dict)
        self.action_type = ActionType[action_spec['type']]
        self.action_spec = action_spec
        self.obs_spec = obs_spec

    def aggregate(self, exp_list):
        '''
            Aggregates a list of subtrajectory dictionaries into dictionary of 
            batched sub-trajectory
            Args:
                exp_list: list of dictionaries that are sub-trajectory attribute
            Returns:
                dict of tensorized subtrajectory information
        '''
        observations, next_obs, actions, rewards, dones, persistent_infos, onetime_infos = [], [], [], [], [], [], []
        for exp in exp_list:
            action_n_step, reward_n_step, done_n_step = self._stack_n_step_experience(exp)
            actions.append(action_n_step)
            rewards.append(reward_n_step)
            dones.append(done_n_step)
            observations.append(exp['obs'])
            next_obs.append([exp['obs_next']])

        observations = self._batch_obs(observations)
        next_obs = self._batch_obs(next_obs)

        if self.action_type == ActionType.discrete:
            actions = np.expand_dims(2).astype('int64')
        elif self.action_type is not ActionType.continuous:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))

        onetime_infos, persistent_infos = self._gather_action_infos(exp_list)
        return {'obs': observations,
                'obs_next': next_obs,
                'actions': np.stack(actions),
                'rewards': np.stack(rewards),
                'persistent_infos': persistent_infos,
                'onetime_infos': onetime_infos,
                'dones': np.stack(dones).astype('float32')}

    def _batch_obs(self, traj_list):
        '''
            Helper function that batches a list of observation (nested dictionary)
            into one whole dictionary of batched observation
            Args:
                traj_list: list of nested observation
        '''
        batched_obs = {}
        for modality in self.obs_spec.keys():
            batched_obs[modality] = {}
            for key in self.obs_spec[modality].keys():
                batched_obs[modality][key] = []
                for exp in traj_list: # exp ~= ex['obs']
                    n_step_obs = []
                    for obs in exp: 
                        n_step_obs.append(obs[modality][key])
                    n_step_obs = np.stack(n_step_obs)
                    batched_obs[modality][key].append(n_step_obs)
                batched_obs[modality][key] = np.stack(batched_obs[modality][key])
        return batched_obs
        
    def _stack_n_step_experience(self, experience):
        """
            Stacks n steps into single numpy arrays
            Args:
                experience: (type: dictionary) subtrajectory information
            Returns:
                observations: stacked numpy array for observations in dim 0
                actions: stacked numpy array for actions in dim 0
                rewards: stacked numpy array for rewards in dim 0
                dones: stacked numpy array for termination flag in dim 0
        """
        actions = np.stack(experience['actions'])
        rewards = np.array(experience['rewards'])
        dones = np.array(experience['dones'])
        return actions, rewards, dones

    def _gather_action_infos(self, exp_list):
        """
            Gathers corresponding action informations from partial trajectories
            Args:
                experience: (type: dictionary) subtrajectory information
            Returns:
                persistent_infos: list of batched FloatTensors
                onetime_infos: one batched FloatTensor
        """
        persistent_infos, onetime_infos = None, None

        exists_ones = (len(exp_list[0]['onetime_infos']) > 0)
        exists_pers = (len(exp_list[0]['persistent_infos'][0]) > 0)

        if exists_ones:
            onetime_infos = []
            for _ in range(len(exp_list[0]['onetime_infos'])):
                onetime_infos.append([])
        if exists_pers:
            persistent_infos = []
            for _ in range(len(exp_list[0]['persistent_infos'][0])):
                persistent_infos.append([])

        for exp in exp_list:
            if exists_ones:
                for i in range(len(onetime_infos)):
                    onetime_infos[i].append(exp['onetime_infos'][i])
            if exists_pers:
                for i in range(len(persistent_infos)):
                    one_exp_info = []
                    for info_list in exp['persistent_infos']:
                        one_exp_info.append(info_list[i])
                    persistent_infos[i].append(np.stack(one_exp_info))

        if exists_ones:
            onetime_infos = [np.stack(info) for info in onetime_infos]
        if exists_pers:
            persistent_infos = [np.asarray(infos) for infos in persistent_infos]

        return onetime_infos, persistent_infos

class NstepReturnAggregator():
    """
        Accepts input by ExpSenderWrapperMultiStepMovingWindow
        Sums over the moving window for bootstrap n_step return

        {
            obs = batch_size * observation
            obs_next = batch_size * next_observation
            actions = batch_size * actions,
            rewards = batch_size,
            dones = batch_size,
        }
    """
    def __init__(self, obs_spec, action_spec, gamma):
        U.assert_type(obs_spec, dict)
        U.assert_type(action_spec, dict)
        self.action_type = ActionType[action_spec['type']]
        self.action_spec = action_spec
        self.obs_spec = obs_spec
        self.gamma = gamma
    
    def aggregate(self, exp_list):
        # TODO add support for more diverse obs_spec and action_spec
        """

        Args:
            exp_list:
        
        Returns:
            aggregated experience
        """
        
        obs0, actions, rewards, obs1, dones, num_steps = (
            collections.defaultdict(list), [], [], collections.defaultdict(list), [], [])
        for exp in exp_list:  # dict
            n_step = exp['n_step']
            num_steps.append(n_step)
            for key in exp['obs_arr'][0]:
                obs0[key].append(np.array(exp['obs_arr'][0][key], copy=False))
            actions.append(exp['action_arr'][0])
            cum_reward = 0
            for i, r in enumerate(exp['reward_arr']):
                cum_reward += pow(self.gamma, i) * r
            rewards.append(cum_reward)
            for key in exp['obs_next']:
                obs1[key].append(np.array(exp['obs_next'][key], copy=False))
            dones.append(float(exp['done_arr'][n_step - 1]))
        for obs in [obs0, obs1]:
            for key in obs:
                obs[key] = np.array(obs[key])

        # TODO: convert action to appropriate numpy array
        if self.action_type == ActionType.continuous:
            actions = np.array(actions, dtype=np.float32)
        elif self.action_type == ActionType.discrete:
            actions = np.array(actions, dtype=np.int32)
        else:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))

        return {
            'obs': dict(obs0),
            'obs_next': dict(obs1),
            'actions': np.array(actions),
            'rewards': np.expand_dims(np.array(rewards), axis=1),
            'dones': np.expand_dims(np.array(dones), axis=1),
            'num_steps': np.expand_dims(np.array(num_steps), axis=1),
        }
