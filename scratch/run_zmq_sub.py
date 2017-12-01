import zmq
import time
import pprint
from surreal.distributed import *


def handler_bar(data):
    print('bar', data)

def handler_foo(data):
    print('foo', data)

client = ZmqSubscribeClient(
    host='127.0.0.1',
    port=8001,
    handler=handler_bar,
    topic='bar',
)
client2 = ZmqSubscribeClient(
    host='127.0.0.1',
    port=8001,
    handler=handler_foo,
    topic='foo',
)
client2.listen_loop(False)
client.listen_loop(True)
