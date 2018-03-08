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
        
        visual_obs0, flat_obs0, actions, rewards, visual_obs1, flat_obs1, dones = [], [], [], [], [], [], []
        for exp in exp_list:  # dict
            visual_obs0.append(np.array(exp['obs'][0][0], copy=False))
            flat_obs0.append(np.array(exp['obs'][0][1], copy=False))
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            visual_obs1.append(np.array(exp['obs'][1][0], copy=False))
            flat_obs1.append(np.array(exp['obs'][1][1], copy=False))
            dones.append(float(exp['done']))
        if self.action_type == ActionType.continuous:
            actions = U.to_float_tensor(actions)
        elif self.action_type == ActionType.discrete:
            actions = torch.LongTensor(actions).unsqueeze(1)
        else:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))

        # Check whether the first element of the list is array(None)
        if visual_obs0[0].shape == ():
            visual_obs0 = None
        else:
            visual_obs0 = U.to_float_tensor(visual_obs0)
        if flat_obs0[0].shape == ():
            flat_obs0 = None
        else:
            flat_obs0 = U.to_float_tensor(flat_obs0)
        if visual_obs1[0].shape == ():
            visual_obs1 = None
        else:
            visual_obs1 = U.to_float_tensor(visual_obs1)
        if flat_obs1[0].shape == ():
            flat_obs1 = None
        else:
            flat_obs1 = U.to_float_tensor(flat_obs1)

        return EasyDict(
            obs=(visual_obs0, flat_obs0),
            obs_next=(visual_obs1, flat_obs1),
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
        observations, actions, rewards, dones = [], [], [], []
        for exp in exp_list:
            observation_n_step, action_n_step, reward_n_step, done_n_step = self.stak_n_step_experience(exp)
            observations.append(observation_n_step)
            actions.append(action_n_step)
            rewards.append(reward_n_step)
            dones.append(done_n_step)
        observations = U.to_float_tensor(np.stack(observations))
        if self.action_type == ActionType.continuous:
            actions = U.to_float_tensor(actions)
        elif self.action_type == ActionType.discrete:
            actions = torch.LongTensor(actions).unsqueeze(2)
        else:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))
        rewards = U.to_float_tensor(rewards)
        dones = U.to_float_tensor(dones)
        return EasyDict(obs=observations,
                    actions=actions, 
                    rewards=rewards, 
                    dones=dones,)

    def stak_n_step_experience(self, experience):
        """
            Stacks n steps into single numpy arrays
        """
        observation_arr = np.stack(experience['obs_arr'])
        action_arr = np.stack(experience['action_arr'])
        reward_arr = np.array(experience['reward_arr'])
        done_arr = np.array(experience['done_arr'])
        return observation_arr, action_arr, reward_arr, done_arr

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
        
        obs0, actions, rewards, obs1, dones = [], [], [], [], []
        for exp in exp_list:  # dict
            n_step = exp['n_step']
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
        )
