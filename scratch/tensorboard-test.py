import random
import time
import sys
import threading
import math
import gym
from visual_logger import TensorboardWrapper, TensorboardMonitor

class EnvThread (threading.Thread):
    def __init__(self, groupID, plotID):
        threading.Thread.__init__(self)
        env = gym.make("PongNoFrameskip-v4")
        self.env = TensorboardWrapper(env, 'localhost', groupID, plotID)

    def run(self):
        print("started routine")
        for _ in range(5):
            self.run_env()
        print("finished routine")

    def run_env(self):
        done = False
        self.env.reset()
        while not done:
            random_action = random.randint(0, 5)
            _, _, done, _ = self.env.step(random_action)


if __name__ == '__main__':
    
    # initializing experiments
    TensorboardMonitor.build_experiments('localhost', num_plots_per_group=2)


    # and then initialize agent threads
    threads = []
    for i in range(6):
        plotID = i // 3
        groupID = i % 3
        env_thread = EnvThread(groupID, plotID)
        threads.append(env_thread)

    for i in range(6):
        threads[i].start()

    for t in threads:
        t.join()

    # example usage of learner side
    tb = TensorboardMonitor('localhost')
    for i in range(100): 
        tb.add_scalar_values(['val1', 'val2', 'val3'], [i, 2* i, 3*i])






