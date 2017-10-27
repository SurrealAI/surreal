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
            # ignore queue name
            _, binary = self._client.brpop(self._queue_name)
            self._handler(binary, i)


class RedisClient:
    def __init__(self, host='localhost', port=6379):
        self.client = redis.StrictRedis(host=host, port=port)
        self.pubsub = self.client.pubsub(ignore_subscribe_messages=True)
        self.queue_threads = {}
        self.subscribe_threads = {}

        # delegated method
        self.set = self.client.set
        self.get = self.client.get
        self.mset = self.client.mset
        self.mget = self.client.mget

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
        """
        handler: function takes an incoming msg from the subscribed channel

        Every message read from a PubSub instance will be a dictionary with the following keys.
        type: One of the following: 'subscribe', 'unsubscribe', 'psubscribe',
            'punsubscribe', 'message', 'pmessage'
        channel: The channel [un]subscribed to or the channel a message was published to
        pattern: The pattern that matched a published message's channel.
            Will be None in all cases except for 'pmessage' types.
        data: The message data. With [un]subscribe messages, this value will be
            the number of channels and patterns the connection is currently
            subscribed to. With [p]message messages,
            this value will be the actual published message.
        """
        self.pubsub.subscribe(**{channel: handler})
        t = self.pubsub.run_in_thread(sleep_time=sleep_time)
        self.subscribe_threads[channel] = t
        return t

    def stop_subscribe_thread(self, channel):
        self.subscribe_threads[channel].stop()
