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
a = 0.15

reverbr = Reverberator(channels, g, a, diff_steps, fs)

# fnd delay line lenghts
#based on article
target_dl_lengths = [113, 125, 138, 150, 163, 175, 188, 200]
dl_lengths = [nearest_prime(int(target_dl_lengths[i] * 0.001 * fs)) for i in range(channels)]

#based on mean free path
mfp_dl_lengths = [29*29, 31*31, 35*35, 37*37, 41*41, 43*43, 47*47, 51*51]

#based on main paths
target_mp_dl_lengths = [7654, 7754, 5093, 5193, 2521, 2621, 9570 , 9677] 
mp_dl_lengths = [nearest_prime(target_mp_dl_lengths[i]) for i in range(channels)]

combo_target = [2337, 2558, 2646, 2911, 3263, 5160, 7718, 9614]
combo_dl_lengths = [nearest_prime(combo_target[i]) for i in range(channels)]

#diffuser dl lengths ms
dl_ms_ranges  = [17, 43, 79, 167, 233]
dl_ms_ranges2 = [29, 57, 89, 107]

#mean free path
mfp_dl_ms_ranges = [58, 60, 67, 70]

er_ms = 29 - int(min(mfp_dl_ms_ranges) / (2 * channels))
er_samples = int(er_ms * fs / 1000)
reverbr.configure(mfp_dl_lengths, mfp_dl_ms_ranges, er_samples)

def main():
    fs, audio = read("guitar.wav")

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
    write("mfp_diff_mfp_fnd_wet.wav", fs, output / 2)

    perc_wet = 0.35
    write("mfp_diff_mfp_fnd_mixed.wav", fs, (audio * (1 - perc_wet) + output * perc_wet) / 2)



if __name__ == "__main__":
    main()
