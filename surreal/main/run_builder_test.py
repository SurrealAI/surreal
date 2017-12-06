from easydict import EasyDict

from surreal.main.ddpg import configs
from surreal.model.model_builders import *

import torch
from torch.autograd import Variable

env_config = EasyDict(configs.env_config)
learner_config = EasyDict(configs.learner_config)

torso = TorsoBuilder(
    input_spec=env_config.obs_spec,
    # conv_spec=learner_config.model.conv_spec,
    conv_spec=None,
    mlp_spec=learner_config.model.mlp_spec
)

env_config.action_spec.type = 'gaussian'

head = HeadBuilder(
    output_spec=env_config.action_spec,
)

# input = torch.randn(2, 3, 32, 32)
input = torch.randn(2, 17)
input = Variable(input)

torso_output = torso(input)
head_output = head(torso_output)

print('torso_output:', torso_output.size())
print('head_output', head_output)