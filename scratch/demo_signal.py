import time
import signal


original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, lambda *args: print('shit'))
for i in range(10):
    time.sleep(1)
    print(i)
signal.signal(signal.SIGINT, original_sigint)
