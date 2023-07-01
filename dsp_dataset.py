import os
import random
import numpy as np
from soundfile import read, write
from dsp import freeverb

# Set the paths
x_file = "balloon_burst.wav"
x_dir = "data/freeverb_dataset/x"
p_dir = "data/freeverb_dataset/p"
y_dir = "data/freeverb_dataset/y"

n_samples = 10

# Create the output directory if it doesn't exist
os.makedirs(p_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)

# Define the Freeverb peters range
n_c = 8
c = (
    (1250, 500), # delay mean, stdev
    (0.8, 0.1), # feedback gain mean, stdev
    (0.5, 0.3) # damping mean, stdev
)

n_a = 4
a = (
    (400, 200), # delay mean, stdev
    (0.5, 0.1) # feedback gain mean, stdev
)

# Set the random seed for reproducibility
random.seed(42)

# Load the audio file
file_path = os.path.join(x_dir, x_file)
x, sr = read(file_path)
# Normalize the x to the range [-1, 1]
x /= max(abs(x))
for i in range(n_samples):
    # Apply the Freeverb effect with randomized peters
    cM = np.random.normal(c[0][0], c[0][1], n_c).astype(int)
    ca = np.random.normal(c[1][0], c[1][1], n_c)
    cd = np.random.normal(c[2][0], c[2][1], n_c)
    aM = np.random.normal(a[0][0], a[0][1], n_a).astype(int)
    aa = np.random.normal(a[1][0], a[1][1], n_a)

    # Clip gain params to prevent exploding feedback
    ca = np.clip(ca, 0, 1)
    cd = np.clip(cd, 0, 1)
    aa = np.clip(aa, 0, 1)

    # Process the audio file with the Freeverb effect
    y = freeverb(
        x=x,
        cM=cM,
        ca=ca,
        cd=cd,
        aM=aM,
        aa=aa
    )
    
    # Store the peters in a .txt file
    p_filename = os.path.splitext(x_file)[0] + f"_{i}.txt"
    p_path = os.path.join(p_dir, p_filename)
    p = np.concatenate([cM, ca, cd, aM, aa], dtype=object)
    # print(peters)
    np.savetxt(p_path, p, fmt='%.10f')
    # Save the processed x to a new audio file
    y_filename = os.path.splitext(x_file)[0] + f"_{i}.wav"
    y_path = os.path.join(y_dir, y_filename)
    # Reshape the processed x to have two dimensions
    # y = y[:, np.newaxis]  # Add the channel dimension back
    write(y_path, y, sr)
    print(f"Generated: {i+1}/{n_samples}")
print(f"{n_samples} samples generated.")