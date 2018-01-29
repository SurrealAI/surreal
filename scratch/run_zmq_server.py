import zmq
from surreal.distributed import *


COUNTER = 0


def handler(req):
    global COUNTER
    reply = '{}-{}'.format(req, COUNTER)
    print(reply)
    if req == 'end':
        COUNTER += 1
    return reply


print('host name', U.host_name())
print('server starts')
server = ZmqServer(
    port=8001,
    handler=handler,
)
server.run_loop(block=True)

