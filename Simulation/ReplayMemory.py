import random
from collections import deque


class ReplayMemory(object):

    def __init__(self, capacity):
        # Define a queue with maxlen "capacity"
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        batch_size = min(batch_size, len(self))

        # Randomly select "batch_size" samples and return the selection
        return random.sample(self.memory, batch_size)

    def extend(self, memory):
        self.memory.extend(memory.memory)

    def __len__(self):
        return len(self.memory)  # Return the number of samples currently stored in the memory

