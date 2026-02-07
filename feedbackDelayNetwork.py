import numpy as np
from numpy.typing import NDArray
from scipy.io.wavfile import write

import random as rnd

from delayLine import DelayLine


def householder(N: int) -> NDArray:
    v = np.ones((N, 1), dtype=np.float32)
    return np.eye(N) - 2 * (v @ v.T) / (v.T @ v)


class LPF:
    def __init__(self, a: float) -> None:
        self.a = a
        self.prev = np.float32(0)

    def attenuate(self, sample: np.float32) -> np.float32:
        out = (1 - self.a) * sample + self.a * self.prev
        self.prev = out
        return out


class FND:
    def __init__(self, channels: int, g: float, a: float, fs: int = 44100) -> None:
        self.channels = channels
        self.a = a
        self.g = g
        self.fs = fs

        self.dls = [DelayLine(i) for i in range(channels)]
        self.dl_lengths = [None] * channels

        self.lpfs = [LPF(a) for _ in range(channels)]
    
        self.matrix = householder(channels)

    def configure(self, dl_lengths: list[int]) -> None:
        for i in range(self.channels):
            self.dls[i].configure(dl_lengths[i])

    def process(self, sample: NDArray) -> NDArray:
        delayed = np.empty_like(sample)
        for i in range(self.channels):
            delayed[i] = self.dls[i].read()

        feedback = self.g * self.matrix @ delayed
        attenuated = np.array([self.lpfs[i].attenuate(feedback[i]) for i in range(self.channels)])

        for i in range(self.channels):
            self.dls[i].write(sample[i] + attenuated[i])

        return delayed
