from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size: int):
        """
        Initializes the replay buffer with a specified size.
        """
        self.buffer_size: int = buffer_size
        self.num_experiences: int = 0
        self.buffer: deque = deque()

    def getBatch(self, batch_size: int):
        """
        Randomly sample a batch of experiences of size `batch_size`.
        If the buffer contains fewer than `batch_size` experiences, 
        it will return all experiences.
        """
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self) -> int:
        """
        Returns the maximum size of the buffer.
        """
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        """
        Adds a new experience to the buffer.
        If the buffer is full, the oldest experience is removed.
        """
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()  # Remove the oldest experience
            self.buffer.append(experience)

    def count(self) -> int:
        """
        Returns the number of experiences currently in the buffer.
        If the buffer is full, returns the buffer size.
        """
        return self.num_experiences

    def erase(self):
        """
        Clears the buffer by resetting it to an empty state.
        """
        self.buffer = deque()
        self.num_experiences = 0
