from surreal.distributed import *
import torch


class ShitNet(U.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(4,3)
        self.w2 = torch.nn.Linear(2,5)
        self.w1.weight.data.zero_()
        self.w1.bias.data.zero_()
        self.w2.weight.data.zero_()
        self.w2.bias.data.zero_()

    def update(self, n):
        self.w1.weight.data += 0.1 * n
        self.w2.weight.data -= 0.1 * n

    def show(self):
        print(round(U.to_scalar(self.w1.weight.mean()), 2), '&',
              round(U.to_scalar(self.w2.weight.mean()), 2))

net = ShitNet()
net2 = ShitNet()


puller = ParameterClient(
    host='localhost',
    port=8002,
    module_dict={'net1': net, 'net2': net2}
)

for i in range(10):
    time.sleep(0.3)
    is_fetched, info = puller.fetch_parameter_with_info()
    print(i, 'is_fetched', is_fetched, info)
    net.show()
    net2.show()

