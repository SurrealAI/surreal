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

net = ShitNet()
net2 = ShitNet()


server = ParameterPublisher(
    port=8001,
    module_dict={'net1': net, 'net2': net2}
)
for i in range(10):
    net.update(1)
    net2.update(3)
    server.publish(i, message='yo'+str(i))
    print('sent', i)
    time.sleep(0.3)
