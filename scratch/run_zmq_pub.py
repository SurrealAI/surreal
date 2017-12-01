import zmq
import time
import pprint
from surreal.distributed import *


server = ZmqPublishServer(
    port=8001,
)
for i in range(10):
    time.sleep(0.2)
    server.publish({'yo':i}, 'foo' if i%2 ==0 else 'bar')
