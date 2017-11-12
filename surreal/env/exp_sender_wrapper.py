from .base import Wrapper
from surreal.distributed import ExpSender, RedisClient
from surreal.session import Config, extend_config
from surreal.agent.default_configs import BASE_AGENT_CONFIG


class ExpSenderWrapper(Wrapper):
    # TODO support N-step obs send
    def __init__(self, env, *, learn_config, session_config):
        super().__init__(env)
        learn_config = extend_config(learn_config, BASE_AGENT_CONFIG)
        if 'sender' not in learn_config:
            sender_config = Config(self.default_config())
        else:
            sender_config = learn_config.sender.extend(self.default_config())
        sender_config['queue_name'] = learn_config.replay.name
        self._client = RedisClient(
            host=session_config.redis.replay.host,
            port=session_config.redis.replay.port
        )
        self.sender = ExpSender(self._client, **sender_config)
        self._obs = None  # obs of the current time step

    def default_config(self):
        return {
            'pointers_only': True,
            'save_exp_on_redis': False,
            'max_redis_queue_size': 10000,
            'obs_cache_size': 10000,
        }

    def _reset(self):
        self._obs, info = self.env.reset()
        return self._obs, info

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self.sender.send([self._obs, obs_next], action, reward, done, info)
        self._obs = obs_next
        return obs_next, reward, done, info
