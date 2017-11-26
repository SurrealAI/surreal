import time
import threading
import multiprocessing
from redis import StrictRedis


r = StrictRedis()


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(self.interval)


def test():
    for i in range(10000):
        key = 'BENCHMARK'+str(i)
        r.set(key, '5jxnviuasjh28437sdifpiu87847230854743085')
        r.get(key)
        r.delete(key)


def bench_single():
    for _ in range(3):
        test()

def bench_thread():  # as slow as single
    ts = [threading.Thread(target=test) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]

def bench_process():  # much faster
    ts = [multiprocessing.Process(target=test) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]


with Timer():
    bench_single()
with Timer():
    bench_thread()
with Timer():
    bench_process()
