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


def torch_aggregate(exp_list, obs_spec, action_spec):
    # TODO add support for more diverse obs_spec and action_spec
    """

    Args:
        exp_list:
        obs_spec:
        action_spec:

    Returns:

    """
    U.assert_type(obs_spec, dict)
    U.assert_type(action_spec, dict)
    obs0, actions, rewards, obs1, dones = [], [], [], [], []
    for exp in exp_list:  # dict
        obs0.append(np.array(exp['obs'][0], copy=False))
        actions.append(exp['action'])
        rewards.append(exp['reward'])
        obs1.append(np.array(exp['obs'][1], copy=False))
        dones.append(float(exp['done']))
    action_type = ActionType[action_spec['type']]
    if action_type == ActionType.continuous:
        actions = _obs_concat(actions)
    elif action_type == ActionType.discrete:
        actions = Variable(torch.LongTensor(actions).unsqueeze(1))
    else:
        raise NotImplementedError('action_spec unsupported '+str(action_spec))
    return dict(
        obs=_obs_concat(obs0),
        obs_next=_obs_concat(obs1),
        actions=actions,
        rewards=Variable(torch.FloatTensor(rewards).unsqueeze(1)),
        dones=Variable(torch.FloatTensor(dones).unsqueeze(1)),
    )
