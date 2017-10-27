import time
import threading
from surreal.comm import RedisClient
from .pointer_queue import PointerQueue
from .exp_download_queue import ExpDownloadQueue
from .base import Replay


class UniformReplay(Replay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = []

    def insert(self, exp_dict):
        pass

    def sample(self, batch_i):
        return None

    def start_sample_condition(self):
        return True

    def aggregate_batch(self, exp_list):
        return exp_list
