"""
Aggregate experience tuple into pytorch-ready tensors
"""
import numpy as np
from easydict import EasyDict
import torch
import surreal.utils as U
from surreal.utils.pytorch import GpuVariable as Variable
from surreal.env import ActionType

def _obs_concat(obs_list):
    # convert uint8 to float32, if any
    return Variable(U.to_float_tensor(np.stack(obs_list)))

class SSARConcatAggregator():
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
            actions = _obs_concat(actions)
        elif self.action_type == ActionType.discrete:
            actions = Variable(torch.LongTensor(actions).unsqueeze(1))
        else:
            raise NotImplementedError('action_spec unsupported '+str(self.action_spec))
        return dict(
            obs=_obs_concat(obs0),
            obs_next=_obs_concat(obs1),
            actions=actions,
            rewards=Variable(torch.FloatTensor(rewards).unsqueeze(1)),
            dones=Variable(torch.FloatTensor(dones).unsqueeze(1)),
        )

class StackNAggregator():
    def __init__(self, obs_spec, action_spec):
        U.assert_type(obs_spec, dict)
        U.assert_type(action_spec, dict)
        self.action_type = ActionType[action_spec['type']]
        self.action_spec = action_spec
        self.obs_spec = obs_spec

    def aggregate(self, exp_list):
        """
        returns observation_array, action_array, reward_array, done_array
        discards info
        """
        observation_arrs, action_arrs, reward_arrs, done_arrs = [], [], [], []
        for exp in exp_list:
            observation_arr, action_arr, reward_arr, done_arr = self.clean_up_raw_exp(exp)
            observation_arrs.append(observation_arr)
            action_arrs.append(action_arr)
            reward_arrs.append(reward_arr)
            done_arrs.append(done_arr)
        return dict(observations=np.stack(observation_arrs), 
                    actions=np.stack(action_arrs), 
                    rewards=np.stack(reward_arrs), 
                    dones=np.stack(done_arrs)
                    )

    def clean_up_raw_exp(self, experience):
        n_step = experience['n_step']
        observation_arr = np.stack([experience[str(i)] for i in range(n_step)])
        action_arr = np.stack(experience['action_arr'])
        reward_arr = np.array(experience['reward_arr'])
        done_arr = np.array(experience['done_arr'])
        return observation_arr, action_arr, reward_arr, done_arr
