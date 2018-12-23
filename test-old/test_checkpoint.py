import os
import numpy as np
from surreal.utils.checkpoint import *
import surreal.utils as U

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

torch.manual_seed(42)

DATA_FOLDER = '~/Temp/data'
DATA_FOLDER = U.f_expand(DATA_FOLDER)


def get_loader(train=False):
    return torch.utils.data.DataLoader(
        datasets.MNIST(DATA_FOLDER, train=train, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]), download = True),
        batch_size=12, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


RESTORE_FOLDER = U.f_join(DATA_FOLDER, 'resume-ckpt')
ORIGINAL_FOLDER = U.f_join(DATA_FOLDER, 'original-ckpt')


class Trainer():
    def __init__(self, restore=False):
        self.model = Net()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.steps = 0
        self.epoch = 0
        self._period = 10
        self.checkpoint = PeriodicCheckpoint(
            folder=RESTORE_FOLDER if restore else ORIGINAL_FOLDER,
            name='unittest',
            tracked_obj=self,
            tracked_attrs=['steps', 'epoch', 'model', 'optimizer'],
            keep_history=5,
            keep_best=3,
            # period=self._period,
            period=self._period
        )
        if restore:
            self.checkpoint.restore(target=0,
                                    mode='history',
                                    check_ckpt_exists=True,
                                    restore_folder=ORIGINAL_FOLDER)

    def train(self):
        losses = []
        self.epoch += 1
        test_loader = get_loader(train=False)
        # must checkpoint the last iteration, so we limit iteration
        for (data, target), _ in zip(test_loader, range(10 * self._period)):
            # print(data.mean(), target.sum())
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.steps += 1
            l = loss.data.item()
            self.checkpoint.save(
                score=-l,
                global_steps=self.steps
            )
            if self.steps % self._period == 0:
                losses.append(l)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, self.steps * len(data),
                    len(test_loader.dataset),
                    100. * self.steps / len(test_loader), l)
                )
        return losses


U.f_remove(RESTORE_FOLDER)
U.f_remove(ORIGINAL_FOLDER)
print('original save checkpoint')
trainer_original = Trainer()
for _ in range(2):  # pretrain 2 epochs
    trainer_original.train()

print('restored continue')
trainer_restored = Trainer(restore=True)
losses_restored = trainer_restored.train()
print('original continue')
losses_original = trainer_original.train()

assert np.allclose(np.array(losses_original), np.array(losses_restored))


def assert_metadata_equal(trainer, trainer_restored):
    meta_original = trainer.checkpoint.metadata
    meta_restored = trainer_restored.checkpoint.metadata
    for field in ['best_ckpt_files', 'best_scores', 'global_steps',
                  'history_ckpt_files', 'keep_best', 'keep_history',
                  'tracked_attrs', 'save_counter', 'version']:
        assert meta_original[field] == meta_restored[field]

    ckpts_original = sorted(list(meta_original.ckpt.keys()))
    ckpts_restored = sorted(list(meta_restored.ckpt.keys()))
    assert ckpts_original == ckpts_restored, (ckpts_restored, ckpts_original)
    for ko, kr in zip(ckpts_original, ckpts_restored):
        for field in ['global_steps', 'save_counter', 'score']:
            assert meta_original.ckpt[ko][field] == meta_restored.ckpt[kr][field]


assert_metadata_equal(trainer_original, trainer_restored)
