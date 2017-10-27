import redis
import threading
import itertools
from surreal.utils.common import StoppableThread


class RedisQueueThread(StoppableThread):
    def __init__(self, redis_client, queue_name, handler, **kwargs):
        self._client = redis_client
        self._queue_name = queue_name
        self._handler = handler
        super().__init__(**kwargs)

    def run(self):
        for i in itertools.count():
            if self.is_stopped():
                break
            binary = self._client.brpop(self._queue_name)
            self._handler(binary, i)


class RedisClient:
    def __init__(self, host='localhost', port=6379):
        self.client = redis.StrictRedis(host=host, port=port)
        self.pubsub = self.client.pubsub(ignore_subscribe_messages=True)
        self.queue_threads = {}
        self.subscribe_threads = {}

    def mset(self, data_dict):
        assert isinstance(data_dict, dict)
        return self.client.mset(data_dict)

    def mget(self, key_list):
        assert isinstance(key_list, list)
        return self.client.mget(key_list)

    def push_to_queue(self, queue_name, binary):
        self.client.lpush(queue_name, binary)

    def pull_from_queue_thread(self, queue_name, handler):
        """
        Forks a thread that runs in an infinite loop, listens on a Redis list
        Args:
          queue_name
          handler: does something upon receiving an object
            [binary_data, index] -> None
        """
        t = RedisQueueThread(self.client, queue_name, handler)
        self.queue_threads[queue_name] = t
        t.start()
        return t

    def stop_queue_thread(self, queue_name):
        self.queue_threads[queue_name].stop()

    def publish(self, channel, msg):
        self.client.publish(channel, msg)

    def subscribe_thread(self, channel, handler, sleep_time=0.1):
        self.pubsub.subscribe(channel, handler)
        t = self.pubsub.run_in_thread(sleep_time=sleep_time)
        self.subscribe_threads[channel] = t
        return t

    def stop_subscribe_thread(self, channel):
        self.subscribe_threads[channel].stop()
