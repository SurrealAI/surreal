import numpy as np
import pyarrow as pa
import time
import pickle
from multiprocessing import Process

data = {
i: np.random.randn(500, 500)
for i in range(100)
}

T0 = time.time()
buf = pa.serialize(data).to_buffer()
T1 = time.time()
print(type(buf))
print(buf.size)

restored_data = pa.deserialize(buf)
print('Serialize: {} ms'.format((T1 - T0) * 1000))
print(restored_data[0])

file = pa.MemoryMappedFile.create('abc', buf.size)
file.write(buf)

def f():
    file2 = pa.memory_map('abc')

    buf2 = file2.read_buffer()
    t1 = time.time()
    for i in range(10000):
        restored_data2 = pa.deserialize(buf2)
    t2 = time.time()
    print('Deserialize: {} ms'.format((t2 - t1) * 1000 / 10000))
    print(restored_data2[0])


p = Process(target=f)
p.start()
p.join()

pi = pickle.dumps(data)
t3 = time.time()
data_pi = pickle.loads(pi)
t4 = time.time()
print('Deserialze pickle: {} ms'.format((t4 - t3) * 1000))

