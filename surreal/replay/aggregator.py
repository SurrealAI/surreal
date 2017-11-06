"""
Aggregate experience tuple into pytorch-ready tensors
"""
import numpy as np
from easydict import EasyDict
import torch
import surreal.utils as U
from surreal.utils.pytorch import GpuVariable as Variable


def _obs_concat(obs_list):
    # convert uint8 to float32, if any
    return Variable(U.to_float_tensor(np.stack(obs_list)))


def aggregate_torch(exp_list):
    obses0, actions, rewards, obses1, dones = [], [], [], [], []
    for exp in exp_list:
        obses0.append(np.array(exp['obses'][0], copy=False))
        actions.append(exp['action'])
        rewards.append(exp['reward'])
        obses1.append(np.array(exp['obses'][1], copy=False))
        dones.append(float(exp['done']))
    return EasyDict(
        obses=[_obs_concat(obses0), _obs_concat(obses1)],
        actions=Variable(torch.LongTensor(actions).unsqueeze(1)),
        rewards=Variable(torch.FloatTensor(rewards).unsqueeze(1)),
        dones=Variable(torch.FloatTensor(dones).unsqueeze(1)),
    )
