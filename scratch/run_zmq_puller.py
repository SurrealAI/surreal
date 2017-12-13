from scratch.utils import *
from surreal.distributed import *
import weakref


# q = ZmqQueue(port=8001,
#              max_size=5,
#              start_thread=True,
#              is_pyobj=False)
#
# i = 0
# while True:
#     i += 1
#     print(U.binary_hash(q.dequeue()), i)
    # time.sleep(0.3)

# class Shit:
#     pass
#
# i=Shit()
# p=[Shit(),3]
# j=Shit()
# print(sys.getrefcount(p[0]))
# print(sys.getrefcount(j))
# print([sys.getrefcount(a)-3 for a in [i,j]])
# print(sys.getrefcount(i))
# print(sys.getrefcount(j))
# sys.exit(0)


mem = []

def exp_tuple_handler(exp_tuple):
    mem.append(exp_tuple)


queue = ExpQueue(
    port=8001,
    max_size=5,
    exp_handler=exp_tuple_handler
)
queue.start_enqueue_thread()
t = queue.start_dequeue_thread()
time.sleep(2)


print(len(mem))
U.print_(queue.weakref_counts())
U.print_(dict(queue._weakref_map))

while mem:
    p = mem.pop()
    print('popped', p)
    print(sys.getrefcount(p['obs'][0]), sys.getrefcount(p['obs'][1]))
    del p
    print(queue.weakref_size(), queue.weakref_keys())
