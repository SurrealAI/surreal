from surreal.utils import *
from collections import deque


N=1000000

d = deque()
with Timer():
    for _ in range(N):
        d.append('iosajfiohbw')
    for _ in range(N):
        d.popleft()


q = FlushQueue(5)
# with Timer():
#     for _ in range(N):
#         q.put('iosajfiohbw')
#     for _ in range(N):
#         q.get()
# sys.exit(0)


def pusher():
    for i in range(30):
        q.put('obj'+str(i))
        time.sleep(0.1)
    print('pusher sleep')
    time.sleep(3)
    for i in range(30, 60):
        q.put('obj'+str(i))
        time.sleep(0.1)


def puller():
    while True:
        print('GET', q.get())
        time.sleep(0.3)


threading.Thread(target=pusher).start()
threading.Thread(target=puller).start()


