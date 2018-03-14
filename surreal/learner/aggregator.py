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
        Accepts input by ExpSenderWrapperMultiStep
        aggregate() returns float Tensors
        EasyDict{
            obs = batch_size * n_step * observation
            next_obs = batch_size * 1 * next_observation
            actions = batch_size * n_step * actions,
            rewards = batch_size * n_step,
            actoin_infos = list of batched FloatTensor action infos
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
        observations, next_obs, actions, rewards, dones, action_infos = [], [], [], [], [], []
        for _ in range(len(exp_list[0]['action_info_arr'][0])):
            action_infos.append([])

        for exp in exp_list:
            observation_n_step, action_n_step, reward_n_step, done_n_step = self.stack_n_step_experience(exp)
            observations.append(observation_n_step)
            actions.append(action_n_step)
            rewards.append(reward_n_step)
            dones.append(done_n_step)
            next_obs.append(exp['obs_next'])

            for i in range(len(action_infos)):
                one_exp_info = []
                for info_list in exp['action_info_arr']:
                    one_exp_info.append(info_list[i])
                action_infos[i].append(np.stack(one_exp_info))

        for i in range(len(action_infos)):
            action_infos[i] = U.to_float_tensor(np.stack(action_infos[i]))

        action_infos = [U.to_float_tensor(np.asarray(infos)) for infos in action_infos]
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
                    action_infos=action_infos,
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
