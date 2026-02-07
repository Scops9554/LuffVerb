import numpy as np
from numpy.typing import NDArray
from random import uniform


class DelayLine:
    def __init__(self, id: int, length: int = 0) -> None:
        self.id = id
        self.dl = np.zeros(length, dtype=np.float32)
        self.pointer = 0
        self.length = length

    def configure(self, length: int) -> None:
        self.dl = np.zeros(length)
        self.length = length
        self.pointer = 0

    def read(self) -> np.float32:
        return self.dl[self.pointer]

    def write(self, sample: np.float32) -> None:
        self.dl[self.pointer] = sample
        self.pointer = (self.pointer + 1) % self.length
