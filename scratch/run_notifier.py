from surreal.utils.tmux import *
from surreal.comm import *
from surreal.replay import *
from surreal.ps import *

client = RedisClient()
notifier = PSNotifier(client, 'ps')

for i in range(80):
    notifier.update('NN '+str(i), i*10)
    sleep(0.1)
