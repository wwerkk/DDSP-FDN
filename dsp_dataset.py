import os
import random
import numpy as np
from soundfile import read, write
from dsp import freeverb

# Set the paths
dry_dir = "data/freeverb_dataset/dry"
param_dir = "data/freeverb_dataset/params"
wet_dir = "data/freeverb_dataset/wet"
filename = "balloon_burst.wav"

n_samples = 10

# Create the output directory if it doesn't exist
os.makedirs(param_dir, exist_ok=True)
os.makedirs(wet_dir, exist_ok=True)

# Define the Freeverb parameters range
n_c = 8
c = (
    (100., 6000.), # delay time range
    (0.1, 0.9), # feedback gain gain
    (0.1, 0.9) # damping range
)

n_a = 4
a = (
    (100, 6000),
    (0.1, 0.9)
)

# Set the random seed for reproducibility
random.seed(42)

# Load the audio file
file_path = os.path.join(dry_dir, filename)
waveform, sample_rate = read(file_path)
# waveform = waveform.to(float32)  # Convert to float32 if necessary


c_delays = np.array([1116, 1617, 1491, 1422, 1277, 1356, 1188, 1116])
c_gains = np.array([0.84 for i in range(8)])
c_damps = np.array([0.2 for i in range(8)])
a_delays = np.array([556, 441, 341, 225])
a_gains = np.array([0.5 for i in range(4)])

# Normalize the waveform to the range [-1, 1]
waveform /= max(abs(waveform))
for i in range(n_samples):
    # Apply the Freeverb effect with randomized parameters
    # c_delays = np.random.randint(int(c[0][0]), int(c[0][1]), size=n_c)
    # c_gains = np.random.uniform(c[1][0], c[1][1], n_c)
    # c_damps = np.random.uniform(c[2][0], c[2][1], n_c)
    # a_delays = np.random.randint(int(a[0][0]), int(a[0][1]), size=n_a)
    # a_gains = np.random.uniform(a[1][0], a[1][1], n_a)

    # Process the audio file with the Freeverb effect
    processed_waveform = freeverb(
        input_signal=waveform,
        c_delays=c_delays,
        c_gains=c_gains,
        c_damps=c_damps,
        a_delays=a_delays,
        a_gains=a_gains
    )
    # processed_waveform = freeverb(waveform)
    
    # Store the parameters in a .txt file
    param_filename = os.path.splitext(filename)[0] + f"_{i}.txt"
    param_path = os.path.join(param_dir, param_filename)
    parameters = np.concatenate([c_delays, c_gains, c_damps, a_delays, a_gains], dtype=object)
    # print(parameters)
    np.savetxt(param_path, parameters, fmt='%.5f')
    # Save the processedt56 waveform to a new audio file
    wet_filename = os.path.splitext(filename)[0] + f"_{i}.wav"
    wet_path = os.path.join(wet_dir, wet_filename)
    # Reshape the processed waveform to have two dimensions
    # processed_waveform = processed_waveform[:, np.newaxis]  # Add the channel dimension back
    write(wet_path, processed_waveform, sample_rate)
    print(f"Generated: {i+1}/{n_samples}")
print(f"{n_samples} samples generated.")