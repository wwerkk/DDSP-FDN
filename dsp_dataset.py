import os
import random
import numpy as np
from soundfile import read, write
from dsp_ import freeverb

# Set the paths
dry_dir = "data/freeverb_dataset/dry"
param_dir = "data/freeverb_dataset/params"
wet_dir = "data/freeverb_dataset/wet"
filename = "noise_burst.wav"

n_samples = 2000

# Create the output directory if it doesn't exist
os.makedirs(param_dir, exist_ok=True)
os.makedirs(wet_dir, exist_ok=True)

# Define the Freeverb parameters range in format (mean, deviaiton)
# Adjust the mean an deviation accor ing to your desired parameter space
n_combs = 8
comb_delays_d = (100, 2000)
comb_gains_d = (0.1 , 0.9)
n_allpasses = 4
allpass_delays_d = (100, 2000)
allpass_gains_d = (0.1, 0.9)

# Set the random seed for reproducibility
random.seed(42)

# Load the audio file
file_path = os.path.join(dry_dir, filename)
waveform, sample_rate = read(file_path)
# waveform = waveform.to(float32)  # Convert to float32 if necessary

# Normalize the waveform to the range [-1, 1]
waveform /= max(abs(waveform))
for i in range(n_samples):
    # Apply the Freeverb effect with randomized parameters
    comb_delays = np.random.randint(comb_delays_d[0], comb_delays_d[1], size=n_combs)
    comb_gains = np.random.uniform(comb_gains_d[0], comb_gains_d[1], n_combs)
    allpass_delays = np.random.randint(allpass_delays_d[0], allpass_delays_d[1], size=n_allpasses)
    allpass_gains = np.random.uniform(allpass_gains_d[0], allpass_gains_d[1], n_allpasses)

    # Process the audio file with the Freeverb effect
    processed_waveform = freeverb(waveform, comb_delays, comb_gains, allpass_delays, allpass_gains)
    # processed_waveform = freeverb(waveform)
    
    # Store the parameters in a .txt file
    param_filename = os.path.splitext(filename)[0] + f"_{i}.txt"
    param_path = os.path.join(param_dir, param_filename)
    parameters = np.concatenate([comb_delays, comb_gains, allpass_delays, allpass_gains], dtype=object)
    # print(parameters)
    np.savetxt(param_path, parameters, fmt='%.5f')
    # Save the processedt56 waveform to a new audio file
    wet_filename = os.path.splitext(filename)[0] + f"_{i}.wav"
    wet_path = os.path.join(wet_dir, wet_filename)
    # Reshape the processed waveform to have two dimensions
    # processed_waveform = processed_waveform[:, np.newaxis]  # Add the channel dimension back
    write(wet_path, processed_waveform, sample_rate)
    print(f"Generated: {i+1}/{n_samples}")
print(f"{len(os.listdir(dry_dir))} samples generated.")