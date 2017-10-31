from .pack import NumpyPack


class ExpPack(NumpyPack):
    unpack_init = True

    def __init__(self, obs_pointers, action, reward, done, info):
        assert isinstance(obs_pointers, list)
        assert isinstance(reward, (float, int))
        assert isinstance(info, dict)
        super().__init__({
            'obs_pointers': obs_pointers,
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        })

    def get_key(self):
        return 'exp:' + super().get_key()


class ObsPack(NumpyPack):
    unpack_init = False

    def get_key(self):
        return 'obs:' + super().get_key()
