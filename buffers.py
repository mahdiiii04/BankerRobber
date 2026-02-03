import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        idx = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in idx]

class ReservoirBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.n_seen = 0  

    def add(self, transition):
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            
            r = np.random.randint(0, self.n_seen)
            if r < self.capacity:
                self.buffer[r] = transition

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        idx = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in idx]
