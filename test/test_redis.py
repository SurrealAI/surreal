from test.utils import *
import torch


class MyModel(U.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Parameter(torch.zeros((3, 2)))
        self.fc2 = torch.nn.Parameter(torch.zeros((5, 3)))

    def add_value(self, x):
        for p in self.parameters():
            p.data.add_(x)

    def get_value(self):
        return U.to_scalar((self.fc1.mean() +self.fc2.mean())/2.)


def test_ps():
    mlearner = MyModel()
    magent = MyModel()
    client = RedisClient()
    client.flushall()
    pslearner = ParameterServer(
        redis_client=client,
        module_dict={'mymodel': mlearner},
        name='yoyoyo'
    )
    psagent = ParameterServer(
        redis_client=client,
        module_dict={'mymodel': magent},
        name='yoyoyo'
    )
    assert psagent.pull() is False  # no PS at all yet

    pslearner.push(iteration=10)
    assert psagent.pull() is True
    assert psagent.pull() is False
    assert psagent.pull() is False

    mlearner.add_value(1)
    pslearner.push(iteration=11)
    assert psagent.pull() is True
    assert psagent.pull() is False
    assert psagent.pull() is False
    assert magent.get_value() == 1.0

    pslearner.push(iteration=12)
    assert psagent.pull() is False
    assert magent.get_value() == 1.0

    mlearner.add_value(1)
    pslearner.push(iteration=13)
    mlearner.add_value(1)
    pslearner.push(iteration=14)
    assert magent.get_value() == 1.0
    assert psagent.pull() is True
    assert psagent.pull() is False
    assert magent.get_value() == 3.0
