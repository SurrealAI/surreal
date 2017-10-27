import torch
import torch.distributed as dist
import argparse
from time import sleep


parser = argparse.ArgumentParser()
parser.add_argument('rank', type=int)
args = parser.parse_args()

dist.init_process_group(backend='tcp', init_method='tcp://127.0.0.1:8000', rank=args.rank, world_size=4)

mygroup = dist.new_group([0, 1, 3])

r = dist.get_rank()

if r == 0:
    T = torch.zeros(3, 3) + 1.2
    dist.broadcast(T, src=0, group=mygroup)
    sleep(3)
    print('bd once')
    T += 0.3
    dist.broadcast(T, src=0, group=mygroup)
    sleep(5)
    print('done')
    dist.broadcast(T, src=0, group=mygroup)
elif r in [2]:
    T2 = torch.zeros(3, 3)
elif r in [1, 3]:
    T2 = torch.zeros(3, 3)
    req = dist.recv(T2)
    print('get', T2)
    print('recv', dist.recv(T2))
    print('get2', T2)
    print('recv', dist.recv(T2))
    print('get3', T2)
