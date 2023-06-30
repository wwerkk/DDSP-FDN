import numpy as np
from freeverb_crap import freeverb
import soundfile as sf

# Generate an input signal
input, sr = sf.read('data/iir_dataset/dry/balloon_burst_1.wav')

# Apply Freeverb to the input signal
gain = 0.9
delay = 0.7
decay = 1.5
output = freeverb(input, sr, gain, delay, decay)
sf.write('reverb_test.wav', output, sr)