from numpy.typing import NDArray
import numpy as np
from diffuser import Diffuser
from feedbackDelayNetwork import FND


class Reverberator:
    def __init__(self, channels: int, g: float, a: float, amt_steps: int, fs: int) -> None:
        self.channels = channels
        self.fs = fs

        self.diff = Diffuser(channels, amt_steps, fs)
        self.fnd = FND(channels, g, a, fs)

    def configure(self, dl_lengths: list[int], dl_ms_ranges: list[int]) -> None:
        self.fnd.configure(dl_lengths)
        self.diff.configure(dl_ms_ranges)

    def process(self, sample: np.float32) -> np.float32:
        input = np.full(self.channels, sample)
        input = self.diff.process(input)
        return self.fnd.process(input).mean()

