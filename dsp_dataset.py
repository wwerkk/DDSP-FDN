import os
import numpy as np
from soundfile import read, write
from dsp import freeverb
import time
from librosa.feature import mfcc
from sklearn.preprocessing import scale

# Set the paths
x_file = "balloon_burst.wav"
x_dir = "data/freeverb_dataset/x"
p_dir = "data/freeverb_dataset/p"
p_n_dir = "data/freeverb_dataset/p_n"
mfcc_dir = "data/freeverb_dataset/mfcc"
mfcc_n_dir = "data/freeverb_dataset/mfcc_n"
y_dir = "data/freeverb_dataset/y"

n_samples = 20
length_s = 2

# Create the output directory if it doesn't exist
os.makedirs(p_n_dir, exist_ok=True)
os.makedirs(mfcc_dir, exist_ok=True)
os.makedirs(mfcc_n_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)

# Define the Freeverb parameter space
n_c = 8
c = (
    (1000, 500), # delay mean, stdev
    (0.75, 0.1), # feedback gain mean, stdev
    (0.2, 0.05) # damping mean, stdev
)

n_a = 4
a = (
    (500, 200), # delay mean, stdev
    (0.5, 0.2) # feedback gain mean, stdev
)

# Set the random seed for reproducibility
np.random.seed(42)

# Load the audio file
file_path = os.path.join(x_dir, x_file)
x, sr = read(file_path)
# Normalize the x to the range [-1, 1]
x /= max(abs(x))
length = length_s * sr

start_time = time.time()

P = []
MFCCS = []

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

    # Truncate to desired length
    y = y[:length]

    # Store the params in a .txt file
    p_filename = os.path.splitext(x_file)[0] + f"_{i}.txt"
    p_path = os.path.join(p_dir, p_filename)
    p = np.concatenate([cM, ca, cd, aM, aa], dtype=object)
    P.append(p)
    np.savetxt(p_path, p, fmt='%.10f')

    # Calculate MFCCs from the processed audio
    mfccs = mfcc(y=y, sr=sr, n_mfcc=16, n_fft = 1024, hop_length=256)
    mfccs = mfccs.reshape(mfccs.shape[:-2] + (-1,))
    MFCCS.append(mfccs)
    # Save MFCCs to a .txt file
    mfcc_filename = os.path.splitext(x_file)[0] + f"_{i}.txt"
    mfcc_path = os.path.join(mfcc_dir, mfcc_filename)
    np.savetxt(mfcc_path, mfccs, fmt='%.10f')

    # Save the processed x to a new audio file
    y_filename = os.path.splitext(x_file)[0] + f"_{i}.wav"
    y_path = os.path.join(y_dir, y_filename)
    write(y_path, y, sr)
    print(f"Generated: {i+1}/{n_samples}")

P = np.array(P)
MFCCS = np.array(MFCCS)
print(f"{n_samples} samples generated.")
print(f"{P.shape} parameters calculated.")
print(f"{MFCCS.shape} MFCCS calculated.")

P = scale(P, axis=0)
MFCCS = scale(MFCCS, axis=1)

for i in range(n_samples):
    # Store the normalised params in a .txt file
    p_filename = os.path.splitext(x_file)[0] + f"_{i}.txt"
    p_path = os.path.join(p_n_dir, p_filename)
    np.savetxt(p_path, P[i], fmt='%.10f')
    # Store the normalised MFCCs in a .txt file
    mfcc_filename = os.path.splitext(x_file)[0] + f"_{i}.txt"
    mfcc_path = os.path.join(mfcc_n_dir, mfcc_filename)
    np.savetxt(mfcc_path, MFCCS[i], fmt='%.10f')

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")