from surreal.utils.tmux import *
from surreal.comm import *
from surreal.replay import *
from surreal.ps import *

client = RedisClient()
listener = Listener(client, 'ps')

def updater(binary, msg):
    print('RECV msg', msg, 'UPDATE NN', binary)
    sleep(1.2) # simulate update lag

listener.run_listener_thread(updater)

