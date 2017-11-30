import surreal.utils as U
from time import sleep
import threading

lock = threading.Lock()
job_queue = U.JobQueue(wait_for_result=1)

def compute(s, i):
    print(s)
    return s+str(len(s)*10 + i)

def f():
    for i in range(40):
        print(job_queue.process(compute, 'ff', i))
        sleep(0.5)

def g():
    for i in range(20):
        print(job_queue.process(compute, 'ggg', i))
        sleep(1)

job_queue.start_thread()

t=threading.Thread(target=f)
t.start()
t2=threading.Thread(target=g)
t2.start()
sleep(10)
job_queue.stop_thread()

# t.join()
# t2.join()

# print(job_queue.result_deque())
