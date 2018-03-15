import torch
import torch.nn as nn

# Inspired by https://github.com/pytorch/pytorch/issues/1959
class LayerNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, x):
        # For (N, C, H, W), we want to average across C
        assert len(x.shape) == 4
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return (x - mean) / (std + self.eps)