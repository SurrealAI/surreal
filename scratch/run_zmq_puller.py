from scratch.utils import *


q = ZmqQueue(port=8001,
         max_size=5,
         start_thread=True,
         to_pyobj=False)

i = 0
while True:
    i += 1
    print(U.binary_hash(q.dequeue()), i)
    # time.sleep(0.3)
