from multiprocessing import Process
import time 

class TestProcess(Process):
    def run(self):
        time.sleep(2)
        print('def')
        exit(11)

class TestProcess2(Process):
    def run(self):
        time.sleep(10)
        print('abc')
        return 'abc'


processes = []

for i in range(5):
    p = TestProcess()
    p.start()
    processes.append(p)

for p in processes:
    p.join()
    print('p', p.exitcode)

p = TestProcess2()
p.start()
p.join()

# def start_replay(self, index):
#     replay = self.replay_class(self.learner_config, self.env_config, self.session_config, index=index)
#     replay.start_threads()
#     replay.join()

# def join(self):
#     self.collector_proxy.join()
#     self.sampler_proxy.join()
    
    
