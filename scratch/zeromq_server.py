import zmq
from surreal.distributed import *


def handler(req):
    print(req)
    return req * 5


server = ZmqServer(
    port=8001,
    handler=handler,
)
server.serve_loop(block=True)

