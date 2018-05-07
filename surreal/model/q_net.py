import torch.nn as nn
from torch.nn.init import xavier_uniform
import surreal.utils as U
import torch.nn.functional as F
import numpy as np
import torchx.nn as nnx


class DuelingQbase(nnx.Module):
    def init_dueling(self, *,
                     action_dim,
                     prelinear_size,
                     fc_hidden_sizes,
                     dueling):
        """
        Args:
            - prelinear_size: size of feature vector before the linear layers,
                like flattened conv or LSTM features
            - fc_hidden_sizes: list of fully connected layer sizes before `action_dim` softmax
        """
        self.dueling = dueling
        self.prelinear_size = prelinear_size
        U.assert_type(fc_hidden_sizes, list)
        hiddens = [prelinear_size] + fc_hidden_sizes
        self.fc_action_layers = nn.ModuleList()
        hidden_list = hiddens + [action_dim]
        for prev_h, next_h in zip(hidden_list[:-1], hidden_list[1:]):
            lin = nn.Linear(prev_h, next_h)
            U.conv_fc_init(lin)
            self.fc_action_layers.append(lin)

        if dueling:
            self.fc_state_layers = nn.ModuleList()
            # output a single state value
            hidden_list = hiddens + [1]
            for prev_h, next_h in zip(hidden_list[:-1], hidden_list[1:]):
                lin = nn.Linear(prev_h, next_h)
                U.conv_fc_init(lin)
                self.fc_state_layers.append(lin)

    def forward(self, x):
        """
        x is the processed tensor from raw states before the dueling layers,
        its shape should be [batch x prelinear_size]
        """
        shape = x.size()
        assert len(shape) == 2 and shape[1] == self.prelinear_size

        action_scores = x
        for is_last, fc_action in U.iter_last(self.fc_action_layers):
            action_scores = fc_action(action_scores)
            if not is_last:
                action_scores = F.relu(action_scores)

        if self.dueling:
            action_scores_mean = action_scores.mean(dim=1, keepdim=True)
            action_scores -= action_scores_mean
            state_score = x
            for is_last, fc_action in U.iter_last(self.fc_state_layers):
                state_score = fc_action(state_score)
                if not is_last:
                    state_score = F.relu(state_score)
            return action_scores + state_score
        else:
            return action_scores


class FFQfunc(DuelingQbase):
    def __init__(self, *,
                 input_shape,
                 action_dim,
                 convs,
                 fc_hidden_sizes,
                 dueling,
                 is_uint8=False):
        """
        is_uint8: scale down uint8 image input from Atari by 255.0
        You can use atari_wrapper.ScaledFloatFrame, but dividing on GPU is more
        computationally and memory efficient
        """
        super().__init__()
        if convs:
            assert len(input_shape) == 3
        self.is_uint8 = is_uint8
        shape = input_shape
        in_channel = input_shape[0]
        self.conv_layers = nn.ModuleList()
        for out_channel, kernel_size, stride in convs:
            # MUST pad to make 'SAME' mode (TF default), otherwise learning much worse
            # pytorch defaults to 'VALID' mode, i.e. pad=0
            conv = nn.Conv2d(in_channel,
                             out_channel,
                             kernel_size,
                             stride=stride,
                             padding=kernel_size // 2)
            U.conv_fc_init(conv)
            in_channel = out_channel
            self.conv_layers.append(conv)
            shape = U.infer_shape_conv2d(shape, out_channel,
                                         kernel_size,
                                         stride=stride,
                                         padding=kernel_size // 2)

        if not U.is_valid_shape(shape):
            raise ValueError('Pre-FC shape has invalid size: {}'.format(shape))

        self.init_dueling(action_dim=action_dim,
                          prelinear_size=int(np.prod(shape)),
                          fc_hidden_sizes=fc_hidden_sizes,
                          dueling=dueling)

    def forward(self, x):
        if self.is_uint8:
            # do NOT use .div_, because the observation may be reused later
            x = x / 255.0
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = U.flatten_conv(x)
        return super().forward(x)


def build_ffqfunc(learner_config, env_config):
    action_dim = env_config.action_spec.dim
    assert len(action_dim) == 1
    assert env_config.action_spec.type == 'discrete'
    action_dim = action_dim[0]
    q_func = FFQfunc(
        input_shape=env_config.obs_spec.dim,
        action_dim=action_dim,
        convs=learner_config.model.convs,
        fc_hidden_sizes=learner_config.model.fc_hidden_sizes,
        dueling=learner_config.model.dueling
    )
    return q_func, action_dim
