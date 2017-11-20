import random
import time
import sys
import threading
import math
import gym
from monitorEnv import TensorboardWrapper

class EnvThread (threading.Thread):
    def __init__(self, setID, groupID):
        threading.Thread.__init__(self)
        env = gym.make("PongNoFrameskip-v4")
        self.env = TensorboardWrapper(env, setID, groupID)

        self.setID = setID
        self.groupID = groupID

    def run(self):
        print("started routine")
        for eps in range(10):
            print("Thread_{}_{} running episode {}".format(self.setID, self.groupID, eps + 1))
            self.run_env()
        print("finished routine")

    def run_env(self):
        done = False
        self.env.reset()
        while not done:
            random_action = random.randint(0, 5)
            _, _, done, _ = self.env.step(random_action)


if __name__ == '__main__':
    # and then initialize agent threads
    threads = []
    for i in range(6):
        setID = i // 3
        groupID = i % 3
        # 2 sets, 3 groups. total 6 plots
        env_thread = EnvThread(setID, groupID)
        threads.append(env_thread)

    for i in range(6):
        threads[i].start()

    for t in threads:
        t.join()





