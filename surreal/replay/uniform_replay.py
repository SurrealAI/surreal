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


class ReplayBuffer(object):
    def __init__(self, size, obs_encode_mode='concat'):
        """
        Args:
          size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
          obs_encode_mode: 'concat' or 'seq'
            if concat, assume each new_obs is of the same shape
            if seq, stack the variable-length sequences into
                torch_util.utils.nn.rnn.PackedSequence, assume [seq_len x feature_size]
        """
        # TODO: also drop Redis memory
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        assert obs_encode_mode in ['concat', 'seq']
        self.obs_encode_mode = obs_encode_mode

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _obs_encode(self, obs_list):
        if self.obs_encode_mode == 'concat':
            return _obs_encode_concat(obs_list)
        elif self.obs_encode_mode == 'seq':
            return _obs_encode_seq(obs_list)
        else:
            raise NotImplementedError


    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        encoded_tensors = (self._obs_encode(obses_t),
                           Variable(LongTensor(actions).unsqueeze(1)),
                           Variable(FloatTensor(rewards).unsqueeze(1)),
                           self._obs_encode(obses_tp1),
                           Variable(FloatTensor(dones).unsqueeze(1)))
        return encoded_tensors

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Args:
          batch_size: int
            How many transitions to sample.

        Returns:
          obs_batch: FloatTensor
            batch of observations
          act_batch: FloatTensor
            batch of actions executed given obs_batch
          rew_batch: FloatTensor
            rewards received as results of executing act_batch
          next_obs_batch: FloatTensor
            next set of observations seen after executing act_batch
          done_mask: FloatTensor
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
