from .wrapper import Wrapper
from surreal.session import Config, extend_config, BASE_SESSION_CONFIG
from surreal.distributed.exp_sender import ExpSender


class ExpSenderWrapper(Wrapper):
    # TODO support N-step obs send
    def __init__(self, env, *, session_config):
        """
        Default sender configs are in BASE_SESSION_CONFIG['sender']
        """
        super().__init__(env)
        config = Config(session_config).extend(BASE_SESSION_CONFIG)
        self.sender = ExpSender(
            host=config.replay.host,
            port=config.replay.port,
            flush_iteration=config.sender.flush_iteration,
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
