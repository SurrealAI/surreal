"""
Jupyter interactive redis commands.
"""
from surreal.orchestrate.jupyter.interactive_util import *
import redis
from redis import StrictRedis


_set_redis_var, _get_global_var = create_interactive_suite(
    suite_name='redis',
    var_class=StrictRedis,
)


def set_redis_var(global_dict, var_name):
    _set_redis_var(global_dict, var_name)
    _create_interactive_methods(global_dict)


def _print_all_methods():
    for name, func in sorted(vars(redis.StrictRedis).items()):
        if not name.startswith('_') and callable(func):
            print("'{}',".format(name))


# append '_' to method names that might have a conflict with other things
_CONFLICT_METHOD_NAMES = [
    'eval',
    'set',
    'type',
    'time',
    'cluster',
    'save'
]


REDIS_METHOD_NAMES = [
    'append',
    'bgrewriteaof',
    'bgsave',
    'bitcount',
    'bitop',
    'bitpos',
    'blpop',
    'brpop',
    'brpoplpush',
    'client_getname',
    'client_kill',
    'client_list',
    'client_setname',
    'cluster',
    'config_get',
    'config_resetstat',
    'config_rewrite',
    'config_set',
    'dbsize',
    'debug_object',
    'decr',
    'delete',
    'dump',
    'echo',
    'eval',
    'evalsha',
    'execute_command',
    'exists',
    'expire',
    'expireat',
    'flushall',
    'flushdb',
    'geoadd',
    'geodist',
    'geohash',
    'geopos',
    'georadius',
    'georadiusbymember',
    'get',
    'getbit',
    'getrange',
    'getset',
    'hdel',
    'hexists',
    'hget',
    'hgetall',
    'hincrby',
    'hincrbyfloat',
    'hkeys',
    'hlen',
    'hmget',
    'hmset',
    'hscan',
    'hscan_iter',
    'hset',
    'hsetnx',
    'hstrlen',
    'hvals',
    'incr',
    'incrby',
    'incrbyfloat',
    'info',
    'keys',
    'lastsave',
    'lindex',
    'linsert',
    'llen',
    'lock',
    'lpop',
    'lpush',
    'lpushx',
    'lrange',
    'lrem',
    'lset',
    'ltrim',
    'mget',
    'move',
    'mset',
    'msetnx',
    'object',
    'parse_response',
    'persist',
    'pexpire',
    'pexpireat',
    'pfadd',
    'pfcount',
    'pfmerge',
    'ping',
    'pipeline',
    'psetex',
    'pttl',
    'publish',
    'pubsub',
    'pubsub_channels',
    'pubsub_numpat',
    'pubsub_numsub',
    'randomkey',
    'register_script',
    'rename',
    'renamenx',
    'restore',
    'rpop',
    'rpoplpush',
    'rpush',
    'rpushx',
    'sadd',
    'save',
    'scan',
    'scan_iter',
    'scard',
    'script_exists',
    'script_flush',
    'script_kill',
    'script_load',
    'sdiff',
    'sdiffstore',
    'sentinel',
    'sentinel_get_master_addr_by_name',
    'sentinel_master',
    'sentinel_masters',
    'sentinel_monitor',
    'sentinel_remove',
    'sentinel_sentinels',
    'sentinel_set',
    'sentinel_slaves',
    'set',
    'set_response_callback',
    'setbit',
    'setex',
    'setnx',
    'setrange',
    'shutdown',
    'sinter',
    'sinterstore',
    'sismember',
    'slaveof',
    'slowlog_get',
    'slowlog_len',
    'slowlog_reset',
    'smembers',
    'smove',
    'sort',
    'spop',
    'srandmember',
    'srem',
    'sscan',
    'sscan_iter',
    'strlen',
    'substr',
    'sunion',
    'sunionstore',
    'time',
    'touch',
    'transaction',
    'ttl',
    'type',
    'unwatch',
    'wait',
    'watch',
    'zadd',
    'zcard',
    'zcount',
    'zincrby',
    'zinterstore',
    'zlexcount',
    'zrange',
    'zrangebylex',
    'zrangebyscore',
    'zrank',
    'zrem',
    'zremrangebylex',
    'zremrangebyrank',
    'zremrangebyscore',
    'zrevrange',
    'zrevrangebylex',
    'zrevrangebyscore',
    'zrevrank',
    'zscan',
    'zscan_iter',
    'zscore',
    'zunionstore',
]


def _create_interactive_methods(global_dict):
    name_pairs = []
    for name in _CONFLICT_METHOD_NAMES:
        name_pairs.append((name, name+'_'))
    for name in set(REDIS_METHOD_NAMES) - set(_CONFLICT_METHOD_NAMES):
        name_pairs.append((name, name))
    for old_name, new_name in name_pairs:
        def _method(*args, __name=old_name, **kwargs):
            return getattr(_get_global_var(), __name)(*args, **kwargs)
        old_method = getattr(StrictRedis, old_name)
        doc = inspect.getdoc(old_method)
        if doc is not None:
            sig = str(inspect.signature(old_method)).replace('(self, ', '(')
            _method.__doc__ = 'signature: ' + sig + '\n' + doc
        global_dict[new_name] = _method
    # special methods
    global_dict['ploadsget'] = lambda key: ploads(get(key))
