import zmq
import time
import pprint
from surreal.distributed import *


client = ZmqClient(
    host='127.0.0.1',
    port=8001,
)
for i in range(10):
    time.sleep(0.2)
    print(client.request('yo'+str(i)))
