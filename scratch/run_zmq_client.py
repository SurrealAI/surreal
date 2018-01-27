import zmq
import time
import pprint
from surreal.distributed import *


print('client starts')
client = ZmqClient(
    # host='server',
    host='localhost',
    port=8001,
)
for i in range(10):
    time.sleep(0.2)
    print(client.request('yo'+str(i)))
