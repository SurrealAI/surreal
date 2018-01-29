import zmq
import time
import os
import subprocess
from surreal.distributed import *
import surreal.utils as U


def get_minikube_ip():
    try:
        return subprocess.check_output(['minikube', 'ip']).decode('utf-8').strip()
    except:
        return None

try:
    print('My IP', os.environ['MY_POD_IP'])
except:
    pass

if U.host_name() == 'minikube':
    host = 'localhost'
else:
    host = get_minikube_ip()
    print('minikube IP', get_minikube_ip())


print('client starts')
client = ZmqClient(
    host='myhost0.serversub.default.svc.cluster.local',
    port=80,
)
for i in range(3):
    time.sleep(0.2)
    print(client.request('yo'+str(i)))
print(client.request('end'))
