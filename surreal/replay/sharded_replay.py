from .base import replay_factory
from surreal.distributed import ZmqLoadBalancerThread
from multiprocessing import Process
from threading import Thread
import os


class ShardedReplay(object):
    def __init__(self,
                 learner_config,
                 env_config,
                 session_config,
                 ):
        """
        Args:
            *_config: passed on to replay
        """
        self.learner_config = learner_config
        self.env_config = env_config
        self.session_config = session_config

        self.replay_class = replay_factory(self.learner_config.replay.replay_class)
        
        self.shards = self.learner_config.replay.replay_shards

        self.collector_frontend_port = os.environ['SYMPH_COLLECTOR_FRONTEND_PORT']
        self.collector_backend_port = os.environ['SYMPH_COLLECTOR_BACKEND_PORT']
        self.sampler_frontend_port = os.environ['SYMPH_SAMPLER_FRONTEND_PORT']
        self.sampler_backend_port = os.environ['SYMPH_SAMPLER_BACKEND_PORT']

        self.collector_frontend_add = "tcp://*:{}".format(self.collector_frontend_port)
        self.collector_backend_add = "tcp://*:{}".format(self.collector_backend_port)
        self.sampler_frontend_add = "tcp://*:{}".format(self.sampler_frontend_port)
        self.sampler_backend_add = "tcp://*:{}".format(self.sampler_backend_port)
        

    def launch(self):
        self.collector_proxy = ZmqLoadBalancerThread(in_add=self.collector_frontend_add, 
                                                    out_add=self.collector_backend_add,
                                                    pattern='pull-push')
        self.sampler_proxy = ZmqLoadBalancerThread(in_add=self.sampler_frontend_add,
                                                    out_add=self.sampler_backend_add,
                                                    pattern='router-dealer')

        self.collector_proxy.start()   
        self.sampler_proxy.start()

        processes = []

        print('Starting {} replay shards'.format(self.shards))
        for i in range(self.shards):
            print('Replay {} starting'.format(i))
            p = Process(target=self.start_replay, args=[i])
            p.start()
            processes.append(p)

    def start_replay(self, index):
        replay = self.replay_class(self.learner_config, self.env_config, self.session_config, index=index)
        replay.start_threads()
        replay.join()

    def join(self):
        self.collector_proxy.join()
        self.sampler_proxy.join()
        
        
