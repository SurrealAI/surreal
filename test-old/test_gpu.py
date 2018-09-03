"""
Test torch_gpu_scope and nn.DataParallel
"""
import os
import numpy as np
import surreal.utils as U

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from surreal.utils.pytorch import GpuVariable as Variable
from surreal.utils.pytorch import Module
# from torch.nn import Module


torch.manual_seed(42)

DATA_FOLDER = '~/Temp/data'
DATA_FOLDER = U.f_expand(DATA_FOLDER)


def get_loader(train=True):
    return torch.utils.data.DataLoader(
        datasets.MNIST(DATA_FOLDER, train=train, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]), download=True),
        batch_size=512*3, shuffle=True)


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


class Trainer():
    def __init__(self):
        if 0:  # test raw nn.Module + DataParallel
            self.model = nn.DataParallel(Net(), device_ids=[0,1,3])
            self.model.cuda()
        else:
            self.model = Net()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.steps = 0
        self.epoch = 0

    def train(self):
        losses = []
        self.epoch += 1
        train_loader = get_loader(train=True)
        for data, target in train_loader:
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.steps += 1
            l = loss.data[0]
            if self.steps % 10 == 0:
                losses.append(l)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, self.steps * len(data),
                    len(train_loader.dataset),
                    100. * self.steps / len(train_loader), l)
                )
        return losses


with U.torch_gpu_scope([0, 3, 2]):
    trainer = Trainer()
    for _ in range(20):
        trainer.train()
