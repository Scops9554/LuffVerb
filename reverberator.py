from numpy.typing import NDArray
import numpy as np
from diffuser import Diffuser
from delayLine import DelayLine
from feedbackDelayNetwork import FND


class Reverberator:
    def __init__(self, channels: int, g: float, a: float, amt_steps: int, fs: int) -> None:
        self.channels = channels
        self.fs = fs

        self.diff = Diffuser(channels, amt_steps, fs)
        self.fnd = FND(channels, g, a, fs)

        self.early_dl = DelayLine(amt_steps + 1)

    def configure(self, dl_lengths: list[int], dl_ms_ranges: list[int], er_samples: int) -> None:
        self.fnd.configure(dl_lengths)
        self.diff.configure(dl_ms_ranges)
        self.early_dl.configure(er_samples)

    def process(self, sample: np.float32) -> np.float32:
        inp = np.full(self.channels, sample)
        inp, er_in = self.diff.process(inp)
        er_out = self.early_dl.read()
        self.early_dl.write(er_in.mean())
        return self.fnd.process(inp).mean() + er_out

