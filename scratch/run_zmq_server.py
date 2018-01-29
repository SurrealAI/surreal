import os
import zmq
import socket
from surreal.distributed import *


print('My IP', os.environ['MY_POD_IP'], 'node', os.environ['MY_NODE_NAME'])
COUNTER = 0


def handler(req):
    global COUNTER
    reply = '{}-{}'.format(req, COUNTER)
    print(reply)
    if req == 'end':
        COUNTER += 1
    return reply


print('host name', U.host_name(), 'FQDN:', socket.getfqdn())
print('server starts')
server = ZmqServer(
    port=8001,
    handler=handler,
)
server.run_loop(block=True)

