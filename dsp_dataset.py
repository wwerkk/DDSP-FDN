import os
import random
import numpy as np
from soundfile import read, write
from dsp import freeverb

# Set the paths
dry_dir = "data/freeverb_dataset/dry"
param_dir = "data/freeverb_dataset/params"
wet_dir = "data/freeverb_dataset/wet"

# Create the output directory if it doesn't exist
os.makedirs(param_dir, exist_ok=True)
os.makedirs(wet_dir, exist_ok=True)

# Define the Freeverb parameters range in format (mean, deviaiton)
# Adjust the mean an deviation according to your desired parameter space
n_combs = 8
comb_delays_d = (1050, 1000)
comb_gains_d = (0.4, 0.2)
n_allpasses = 8
allpass_delays_d = (450, 350)
allpass_gains_d = (0.6, 0.3)

# Set the random seed for reproducibility
random.seed(42)

# Iterate over the audio files in the input directory
for i, filename in enumerate(os.listdir(dry_dir)):
    if filename.endswith(".wav"):
        # Load the audio file
        file_path = os.path.join(dry_dir, filename)
        waveform, sample_rate = read(file_path)
        # waveform = waveform.to(float32)  # Convert to float32 if necessary

        # Normalize the waveform to the range [-1, 1]
        waveform /= max(abs(waveform))

        # Apply the Freeverb effect with randomized parameters
        comb_delays = [int(random.normalvariate(comb_delays_d[0], comb_delays_d[1])) for j in range(n_combs)]
        comb_gains = [random.normalvariate(comb_gains_d[0], comb_gains_d[1]) for i in range(n_combs)]
        allpass_delays = [int(random.normalvariate(allpass_delays_d[0], allpass_delays_d[1])) for j in range(n_allpasses)]
        allpass_gains = [random.normalvariate(allpass_gains_d[0], allpass_gains_d[1]) for j in range(n_allpasses)]

        # Process the audio file with the Freeverb effect
        processed_waveform = freeverb(waveform, comb_delays, comb_gains, allpass_delays, allpass_gains)
        # processed_waveform = freeverb(waveform)
        
        # Store the parameters in a .txt file
        param_filename = os.path.splitext(filename)[0] + f".txt"
        param_path = os.path.join(param_dir, param_filename)
        parameters = np.array([comb_delays, comb_gains, allpass_delays, allpass_gains])
        # print(parameters)
        np.savetxt(param_path, parameters)
        # Save the processed waveform to a new audio file
        wet_path = os.path.join(wet_dir, filename)
        # Reshape the processed waveform to have two dimensions
        # processed_waveform = processed_waveform[:, np.newaxis]  # Add the channel dimension back
        write(wet_path, processed_waveform, sample_rate)
        print(f"Processed: {os.path.splitext(filename)[0]}")
    print(f"{i+1}/{len(os.listdir(dry_dir))}")