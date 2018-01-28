import zmq
import time
import pprint
import subprocess
from surreal.distributed import *


def get_minikube_ip():
    try:
        return subprocess.check_output(['minikube', 'ip']).decode('utf-8').strip()
    except:
        return None

print(get_minikube_ip())


print('client starts')
client = ZmqClient(
    # host=get_minikube_ip(),
    host='localhost',
    port=8001,
)
for i in range(10):
    time.sleep(0.2)
    print(client.request('yo'+str(i)))
