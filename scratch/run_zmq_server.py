import zmq
from surreal.distributed import *


def handler(req):
    print(req)
    return req * 5


print('host name', U.host_name())
print('server starts')
server = ZmqServer(
    port=8001,
    handler=handler,
)
server.run_loop(block=True)

