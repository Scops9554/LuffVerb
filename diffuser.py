from sympy import nextprime, prevprime
import numpy as np
from numpy.typing import NDArray
from scipy.io.wavfile import write

import random as rnd
from delayLine import DelayLine


def hadamard(N: int) -> NDArray:
    assert N & (N-1) == 0

    H = np.array([1])
    while H.shape[0] < N:
        H = np.block([
            [ H, H],
            [-H, H]
        ])

    return H / np.sqrt(N)


def nearest_prime(N: int) -> int:
    if N <= 2:
        return 2

    next = nextprime(N)
    prev = prevprime(N)

    if next - N < N - prev:
        return next

    return prev


def get_dl_lengths(dl_range_ms: int, channels: int, fs: int) -> list[int]:
    sector = int(dl_range_ms * 0.001 * fs / channels)
    base = int(sector / 2)
    var = int(base / 2)
    return [nearest_prime(base + sector * i + rnd.randint(-var, var))
            for i in range(channels)]


class DiffusionStep:
    def __init__(self, channels: int, fs: int) -> None:
        self.channels = channels
        self.fs = fs

        self.dls = [DelayLine(i) for i in range(1, self.channels + 1)]
        self.dl_range_ms = 50

        self.matrix = hadamard(channels)
        self.flips = [None]

    def configure(self, dl_range_ms: int = 50) -> None:
        polarity = [-1, 1]
        self.flips = np.array([rnd.choice(polarity) for i in range(self.channels)]) 

        self.dl_range_ms = dl_range_ms

        dl_lengths = get_dl_lengths(dl_range_ms, self.channels, self.fs)
        for i in range(self.channels):
            self.dls[i].configure(dl_lengths[i])

    def process(self, sample: NDArray) -> NDArray:
        delayed = np.empty_like(sample)
        for i in range(self.channels):
            delayed[i] = self.dls[i].read()
            self.dls[i].write(sample[i])

        mixed = self.matrix @ delayed 
        return mixed * self.flips

            
class Diffuser:
    def __init__(self, channels: int = 8, amt_steps: int = 5, fs: int = 44100) -> None:
        self.channels = channels
        self.fs = fs

        self.amt_steps = amt_steps
        self.steps = [DiffusionStep(channels, fs) for i in range(amt_steps)]

    def configure(self, dl_ranges_ms: list[int]) -> None:
        for i in range(self.amt_steps):
            self.steps[i].configure(dl_ranges_ms[i])

    def process(self, sample: NDArray) -> NDArray:
        for step in self.steps:
            sample = step.process(sample)
        return sample
       

if __name__ == "__main__":
    diff = Diffuser(amt_steps=5)
    dl_ranges = [48, 96, 192, 384, 768]
    dl_ranges2 = [20, 40, 80, 160]
    diff.configure(dl_ranges)

    fs = 44100
    output_len = int(2 * fs)
    output = np.empty(output_len, dtype=np.float32)

    imp = 5 * np.ones(8, dtype=np.float32)
    imp_len = 1
    for i in range(imp_len):
        output[i] = diff.process(imp).mean()

    silence = np.zeros(8, dtype=np.float32)
    for i in range(imp_len, output_len):
        output[i] = diff.process(silence).mean()

    write("test.wav", fs, output)
