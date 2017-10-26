from surreal.comm import RedisClient
from .exp_downloader import ExpDownloader


class UniformReplay(object):
    def __init__(self, host='localhost', port=6379):
        self.client = RedisClient(host=host, port=port)
        self.downloader = ExpDownloader(self.client)


    def sample(self):
        # TODO make sampler and queue overlapping
        pass