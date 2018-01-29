import zmq
import time
import pprint
import socket
import subprocess
from surreal.distributed import *
import surreal.utils as U


def get_minikube_ip():
    try:
        return subprocess.check_output(['minikube', 'ip']).decode('utf-8').strip()
    except:
        return None

print(get_minikube_ip())
print('DNS', socket.getfqdn())


print('client starts')
client = ZmqClient(
    # host=get_minikube_ip(),
    host='localhost',
    port=8001,
)
for i in range(10):
    time.sleep(0.2)
    print(client.request('yo'+str(i)))
