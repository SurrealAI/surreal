"""
Aggregate experience tuple into pytorch-ready tensors
"""
import numpy as np
from easydict import EasyDict
import torch
import surreal.utils as U
from surreal.env import ActionType

class SSARAggregator():
    """
        Accepts experience sent by SSAR experience senders
        aggregate() returns float tensors:
        TODO: make them Tensors 
        EasyDict{
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
        
        obs0, actions, rewards, obs1, dones = [], [], [], [], []
        for exp in exp_list:  # dict
            obs0.append(np.array(exp['obs'][0], copy=False))
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            obs1.append(np.array(exp['obs'][1], copy=False))
            dones.append(float(exp['done']))
        if self.action_type == ActionType.continuous:
            actions = U.to_float_tensor(actions)
        elif self.action_type == ActionType.discrete:
            actions = torch.LongTensor(actions).unsqueeze(1)
        else:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))
        return EasyDict(
            obs=U.to_float_tensor(obs0),
            obs_next=U.to_float_tensor(obs1),
            actions=actions,
            rewards=U.to_float_tensor(rewards).unsqueeze(1),
            dones=U.to_float_tensor(dones).unsqueeze(1),
        )

class MultistepAggregator():
    """
        Accepts input by ExpSenderWrapperMultiStep
        aggregate() returns float Tensors
        TODO: make them Tensors
        EasyDict{
            obs = batch_size * n_step * observation
            next_obs = batch_size * 1 * next_observation
            actions = batch_size * n_step * actions,
            rewards = batch_size * n_step,
            dones = batch_size * n_step,
        }
    """
    def __init__(self, obs_spec, action_spec):
        U.assert_type(obs_spec, dict)
        U.assert_type(action_spec, dict)
        self.action_type = ActionType[action_spec['type']]
        self.action_spec = action_spec
        self.obs_spec = obs_spec

    def aggregate(self, exp_list):
        observations, next_obs, actions, rewards, dones = [], [], [], [], []
        for exp in exp_list:
            observation_n_step, action_n_step, reward_n_step, done_n_step = self.stack_n_step_experience(exp)
            observations.append(observation_n_step)
            actions.append(action_n_step)
            rewards.append(reward_n_step)
            dones.append(done_n_step)
            next_obs.append(exp['obs_next'])
        observations = U.to_float_tensor(np.stack(observations))
        next_obs     = U.to_float_tensor(np.stack(next_obs)).unsqueeze(1)
        if self.action_type == ActionType.continuous:
            actions = U.to_float_tensor(actions)
        elif self.action_type == ActionType.discrete:
            actions = torch.LongTensor(actions).unsqueeze(2)
        else:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))
        rewards = U.to_float_tensor(rewards)
        dones = U.to_float_tensor(dones)
        return EasyDict(obs=observations,
                    next_obs = next_obs,
                    actions=actions, 
                    rewards=rewards, 
                    dones=dones,)

    def stack_n_step_experience(self, experience):
        """
            Stacks n steps into single numpy arrays
        """
        observation_arr = np.stack(experience['obs_arr'])
        action_arr = np.stack(experience['action_arr'])
        reward_arr = np.array(experience['reward_arr'])
        done_arr = np.array(experience['done_arr'])
        return observation_arr, action_arr, reward_arr, done_arr


class MultistepAggregatorWithInfo():
    """
        Accepts input by ExpSenderWrapperMultiStepMovingWindowWithInfo
        aggregate() returns float Tensors
        EasyDict{
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
                EasyDict of tensorized subtrajectory information
        '''
        observations, next_obs, actions, rewards, dones, persistent_infos, onetime_infos = [], [], [], [], [], [], []
        for exp in exp_list:
            observation_n_step, action_n_step, reward_n_step, done_n_step = self.stack_n_step_experience(exp)
            observations.append(observation_n_step)
            actions.append(action_n_step)
            rewards.append(reward_n_step)
            dones.append(done_n_step)
            next_obs.append(exp['obs_next'])

        observations =  U.to_float_tensor(np.stack(observations))
        next_obs     =  U.to_float_tensor(np.stack(next_obs)).unsqueeze(1)
        if self.action_type == ActionType.continuous:
            actions = U.to_float_tensor(actions)
        elif self.action_type == ActionType.discrete:
            actions = torch.LongTensor(actions).unsqueeze(2)
        else:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))
        rewards = U.to_float_tensor(rewards)
        dones = U.to_float_tensor(dones)

        onetime_infos, persistent_infos = self._gather_action_infos(exp_list)

        return EasyDict(obs=observations,
                    next_obs = next_obs,
                    actions=actions, 
                    rewards=rewards, 
                    persistent_infos=persistent_infos,
                    onetime_infos=onetime_infos,
                    dones=dones,)

    def stack_n_step_experience(self, experience):
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
        observations = np.stack(experience['obs'])
        actions = np.stack(experience['actions'])
        rewards = np.array(experience['rewards'])
        dones = np.array(experience['dones'])
        return observations, actions, rewards, dones

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
            onetime_infos = [U.to_float_tensor(np.stack(info)) 
                                for info in onetime_infos]
        if exists_pers:
            persistent_infos = [U.to_float_tensor(np.asarray(infos)) 
                                    for infos in persistent_infos]

        return onetime_infos, persistent_infos

class NstepReturnAggregator():
    """
        Accepts input by ExpSenderWrapperMultiStepMovingWindow
        Sums over the moving window for bootstrap n_step return

        aggregate() returns float tensors:
        TODO: make them Tensors 
        EasyDict{
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
        
        obs0, actions, rewards, obs1, dones, num_steps = [], [], [], [], [], []
        for exp in exp_list:  # dict
            n_step = exp['n_step']
            num_steps.append(n_step)
            obs0.append(np.array(exp['obs_arr'][0], copy=False))
            actions.append(exp['action_arr'][0])
            cum_reward = 0
            for i, r in enumerate(exp['reward_arr']):
                cum_reward += pow(self.gamma, i) * r
            rewards.append(cum_reward)
            obs1.append(np.array(exp['obs_next'], copy=False))
            dones.append(float(exp['done_arr'][n_step - 1]))
        if self.action_type == ActionType.continuous:
            actions = U.to_float_tensor(actions)
        elif self.action_type == ActionType.discrete:
            actions = torch.LongTensor(actions).unsqueeze(1)
        else:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))
        return EasyDict(
            obs=U.to_float_tensor(obs0),
            obs_next=U.to_float_tensor(obs1),
            actions=actions,
            rewards=U.to_float_tensor(rewards).unsqueeze(1),
            dones=U.to_float_tensor(dones).unsqueeze(1),
            num_steps=U.to_float_tensor(num_steps).unsqueeze(1),
        )
