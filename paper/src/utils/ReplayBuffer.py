import numpy as np

# much much faster than np.random.choice
def choice(arr, size=1, rng=np.random):
    idxs = rng.permutation(len(arr))
    return [arr[i] for i in idxs[:size]]

class ReplayBuffer:
    def __init__(self, buffer_size: int, seed: int):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = {}

        self.random = np.random.RandomState(seed)

    def __len__(self):
        return len(self.buffer)

    def add(self, args):
        self.buffer[self.location] = args
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return choice(self.buffer, batch_size, self.random), []

    # match api with prioritized ER buffer
    def update_priorities(self, idxes, priorities):
        pass
