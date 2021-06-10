from utils.per.data_structures import SumSegmentTree, MinSegmentTree
import numpy as np

class PrioritizedReplayMemory(object):
    def __init__(self, size, alpha=0.6):
        print("Buffer Config", size, alpha)
        super(PrioritizedReplayMemory, self).__init__()
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0


    def add(self, data):
        idx = self._next_idx

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _encode_sample(self, idxes):
        return [self._storage[i] for i in idxes]

    def _sample_proportional(self, batch_size):
        s = self._it_sum.sum(0, len(self._storage) - 1)
        out = []
        while len(out) < batch_size:
            mass = s * np.random.rand()
            idx = self._it_sum.find_prefixsum_idx(mass)
            if idx not in out:
                out.append(idx)

        return out

    def __len__(self):
        return len(self._storage)

    def sample(self, batch_size):
        idxes = self._sample_proportional(batch_size)

        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, idxes

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        priorities = np.abs(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority+1e-5) ** self._alpha
            self._it_min[idx] = (priority+1e-5) ** self._alpha

            self._max_priority = max(self._max_priority, (priority+1e-5))
