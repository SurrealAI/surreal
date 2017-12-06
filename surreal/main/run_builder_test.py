from easydict import EasyDict

from surreal.main.ddpg import configs
from surreal.model.model_builders import *

import torch
from torch.autograd import Variable

env_config = EasyDict(configs.env_config)
learner_config = EasyDict(configs.learner_config)

torso = Torso(
    input_spec=env_config.obs_spec,
    # conv_spec=learner_config.model.conv_spec,
    conv_spec=None,
    mlp_spec=learner_config.model.mlp_spec
)

# input = torch.randn(2, 3, 32, 32)
input = torch.randn(2, 1024)
input = Variable(input)
output = torso(input)

# input = torch.randn(2, 3, 32, 32)
input = torch.randn(2, 1024)
input = Variable(input)
output = torso(input)

# print(output)