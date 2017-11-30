from .wrapper import Wrapper
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG


class ExpSenderWrapper(Wrapper):
    # TODO support N-step obs send
    def __init__(self, env, *, session_config):
        """
        Default sender configs are in BASE_SESSION_CONFIG['sender']
        """
        super().__init__(env)
        config = Config(session_config).extend(BASE_SESSION_CONFIG)
        self._client = RedisClient(
            host=session_config.replay.host,
            port=session_config.replay.port
        )
        self.sender = ExpSender(
            self._client,
            queue_name=config.replay.name,
            pointers_only=config.sender.pointers_only,
            remote_exp_queue_size=config.replay.remote_exp_queue_size,
            remote_save_exp=config.sender.remote_save_exp,
            local_obs_cache_size=config.sender.local_obs_cache_size,
        )
        self._obs = None  # obs of the current time step

    def _reset(self):
        self._obs, info = self.env.reset()
        return self._obs, info

    def _step(self, action):
        obs_next, reward, done, info = self.env.step(action)
        self.sender.send([self._obs, obs_next], action, reward, done, info)
        self._obs = obs_next
        return obs_next, reward, done, info
