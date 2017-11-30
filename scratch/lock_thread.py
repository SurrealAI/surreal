import surreal.utils as U
from time import sleep
import threading
import multiprocessing

lock1 = threading.Lock()
lock2 = multiprocessing.Lock()
lock1 = multiprocessing.Lock()

def compute(s, i):
    print(s, i)
    return s+str(len(s)*10 + i)

def f():
    with lock1:
        for i in range(5):
            compute('yo', 3)
            sleep(0.2)

def g():
    with lock1:
        for i in range(5):
            compute('bar', 7)
            sleep(.2)


MULTI = multiprocessing.Process
MULTI = threading.Thread

t=MULTI(target=f)
t.start()
t2=MULTI(target=g)
t2.start()

t.join()
t2.join()

# print(job_queue.result_deque())
