import numpy as np
from dsp_ import comb, lbcf
import soundfile as sf

# Generate an input signal
x, sr = sf.read('data/comb_dataset/input/dry/balloon_burst_1.wav')

# Apply effect to the input signal
b = 0.9
M = 2000
a = 0.99
y = comb(x=x, b=b, M=M, a=a)
sf.write('balloon_comb.wav', y, sr)