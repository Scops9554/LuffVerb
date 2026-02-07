from diffuser import nearest_prime
from reverberator import Reverberator

from random import uniform

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read


fs = 44100
channels = 8
diff_steps = 4
g = 0.85
a = 0.05

reverbr = Reverberator(channels, g, a, diff_steps, fs)

target_dl_lengths = [113, 125, 138, 150, 163, 175, 188, 200]
dl_lengths = [nearest_prime(int(target_dl_lengths[i] * 0.001 * fs)) for i in range(channels)]

dl_ms_ranges = [20, 40, 80, 160]
reverbr.configure(dl_lengths, dl_ms_ranges)

def main():
    fs, audio = read("piano.wav")

    # Convert to float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

    # If stereo, convert to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    output_len = audio.shape[0]
    output = np.empty(output_len, dtype=np.float32)

    for i in range(output_len):
        # assuming reverbr.process returns an array
        output[i] = reverbr.process(audio[i])

    print("writing")
    write("wet.wav", fs, output)

    perc_wet = 0.05
    write("mixed.wav", fs, audio * (1 - perc_wet) + output * perc_wet)



if __name__ == "__main__":
    main()
