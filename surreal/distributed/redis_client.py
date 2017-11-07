import redis
import time
import itertools
import surreal.utils as U


# ====== Lua scripts for Redis ======
# incr many keys together, should be the same as pipelining
_LUA_MINCR = """
local retval
retval={}
for i = 1,#KEYS do
    table.insert(retval, redis.call('incr', KEYS[i]))
end
return retval
"""

# decr many keys together, return values will never drop below 0
# the key is automatically deleted if it drops to 0.
_LUA_MDECR = """
local retval, counts
retval={}
for i = 1,#KEYS do
    counts = redis.call('decr', KEYS[i])
    if counts <= 0 then
        redis.call('del', KEYS[i])
        counts = 0
    end
    table.insert(retval, counts)
end
return retval
"""

# push to the list only when it has enough capacity, otherwise return -1
# implements `BLPUSH` if combined with sleep() spin lock
_LUA_BLPUSH = """
local max_size, num_elems
max_size = tonumber(ARGV[1])
table.remove(ARGV, 1)
num_elems = #ARGV
if redis.call('LLEN', KEYS[1]) + num_elems <= max_size then
    return redis.call('LPUSH', KEYS[1], unpack(ARGV))  
end 
return -1
"""


class _DequeueThread(U.StoppableThread):
    def __init__(self, redis_client, queue_name, handler, **kwargs):
        """
        Args:
            redis_client:
            queue_name:
            handler:
            **kwargs:
        """
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


class RedisClient(object):
    def __init__(self, host='localhost', port=6379):
        self._client = redis.StrictRedis(host=host, port=port)
        self._pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        self._queue_threads = {}
        self._subscribe_threads = {}

        # delegated method
        self.set = self._client.set
        self.get = self._client.get
        self.flushall = self._client.flushall
        self.mincr = self._client.register_script(_LUA_MINCR)
        self.mdecr = self._client.register_script(_LUA_MDECR)
        self._blpush = self._client.register_script(_LUA_BLPUSH)

    def mset(self, mset_dict):
        U.assert_type(mset_dict, dict)
        if len(mset_dict) == 0:
            return False
        else:
            return self._client.mset(mset_dict)

    def mget(self, mget_list):
        U.assert_type(mget_list, list)
        if len(mget_list) == 0:
            return []
        else:
            return self._client.mget(mget_list)

    def mdel(self, mdel_list):
        """
        mass delete keys
        """
        U.assert_type(mdel_list, list)
        if len(mdel_list) == 0:
            return 0
        else:
            return self._client.delete(*mdel_list)

    def blpush(self,
               queue_name,
               values,
               max_size,
               sleep_interval=0.1,
               time_out=0):
        """
        Simulates BLPUSH, block until a Redis list has enough capacity.
        same as Python's synchronized `Queue.put()` semantics.

        Args:
            queue_name:
            values:
            max_size:
            sleep_interval: in seconds, for spin lock
            time_out: if 0, wait indefinitely, otherwise return None if timeout

        Returns:
            queue size (i.e. llen) after pushing
        """
        if not isinstance(values, list):
            values = [values]
        assert max_size > len(values)
        start_time = time.time()
        while True:
            ret = self._blpush(keys=[queue_name], args=[max_size]+values)
            if ret < 0:
                time.sleep(sleep_interval)
                if time_out > 0 and time.time() - start_time > time_out:
                    return None
            else:
                return ret

    enqueue_block_on_full = blpush

    def enqueue(self, queue_name, values):
        """
        Warnings:
            This method does not block when the queue is full. If the agent
            produces exp faster than the Replay can consume, the memory will
            grow indefinitely.
            Use `blpush()` or alias `enqueue_block_on_full()` instead.
        """
        if not isinstance(values, list):
            values = [values]
        self._client.lpush(queue_name, *values)


    def start_dequeue_thread(self, queue_name, handler):
        """
        Forks a thread that runs in an infinite loop, listens on a Redis list
        Args:
          queue_name
          handler: does something upon receiving an object
            [binary_data, index] -> None
        """
        if queue_name in self._queue_threads:
            raise RuntimeError('Queue thread [{}] is already running'
                               .format(queue_name))
        t = _DequeueThread(self._client, queue_name, handler)
        self._queue_threads[queue_name] = t
        t.start()
        return t

    def stop_dequeue_thread(self, queue_name):
        self._queue_threads[queue_name].stop()

    def publish(self, channel, msg):
        self._client.publish(channel, msg)

    def start_subscribe_thread(self, channel, handler, sleep_time=0.1):
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
        self._pubsub.subscribe(**{channel: handler})
        t = self._pubsub.run_in_thread(sleep_time=sleep_time)
        self._subscribe_threads[channel] = t
        return t

    def stop_subscribe_thread(self, channel):
        self._subscribe_threads[channel].stop()
