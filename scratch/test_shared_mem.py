import numpy as np
import pyarrow as pa
import time
import pickle
from multiprocessing import Process
import os

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
file.close()

def f():
    
    t1 = time.time()
    file2 = pa.memory_map('abc')
    buf2 = file2.read_buffer()
    for i in range(100):
        restored_data2 = pa.deserialize(buf2)
        file2.close()
    t2 = time.time()
    print('Deserialize: {} ms'.format((t2 - t1) * 1000 / 100))
    print(restored_data2[0])

    t_delete = time.time()
    os.remove('abc')
    t_delete = time.time() - t_delete
    print('t_delete: {}'.format(t_delete))


p = Process(target=f)
p.start()
p.join()

pi = pickle.dumps(data)
t3 = time.time()
data_pi = pickle.loads(pi)
t4 = time.time()
print('Deserialze pickle: {} ms'.format((t4 - t3) * 1000))

