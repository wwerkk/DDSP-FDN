# Latest 09/11/2023 19:07

# import os
# from tarfile import NUL
import numpy as np
import scipy
import pandas as pd
import soundfile as sf
from soundfile import read, write
from dsp_ import prime
import time
from librosa.feature import mfcc
from dsp_ import simple_fdn

# Set the paths
mfcc_dir = "data/fdn_dataset/mfcc"
mfcc_n_dir = "data/fdn_dataset/mfcc_n"
y_dir = "data/fdn_dataset/y"

# FDN size and samplerate
FDN_SIZE = 16
SAMPLE_RATE = 44100
IMPULSE_NUM = 3

# Set the random seed for reproducibility
np.random.seed(42)

# Unipolar impulse
x = np.zeros(2)
x[0] = -1
x[1] = 1

# Hadamard matrix
H = scipy.linalg.hadamard(FDN_SIZE) * 0.25
start_time = time.time()
MFCCS = []

# Random parameter arrays
decay           = np.random.random((IMPULSE_NUM))
min_dist        = np.random.random((IMPULSE_NUM)) * 0.1
max_dist        = 0.1 + np.random.random((IMPULSE_NUM)) * 0.9
distance_curve  = np.random.random((IMPULSE_NUM))
min_freq        = np.random.random((IMPULSE_NUM)) * 0.5
max_freq        = 0.5 + np.random.random((IMPULSE_NUM)) * 0.5
frequency_curve = np.random.random((IMPULSE_NUM))

parameters = pd.DataFrame(
    {
        "Decay":           decay,
        "Min distance":    min_dist,
        "Max distance":    max_dist,
        "Distance curve":  distance_curve,
        "Low frequency":   min_freq,
        "High frequency":  max_freq,
        "Frequency curve": frequency_curve
    }
)

# prime number list for delay line
PRIME_LIST = prime(0, 30000)

# 2 second impulse responses
IMPULSE_LENGTH = (SAMPLE_RATE * 2) + x.shape[-1]

# store impulse responses in a 2d array.
impulse_responses = np.ndarray((IMPULSE_NUM, IMPULSE_LENGTH))

for i in range(IMPULSE_NUM):
    y = simple_fdn(x,
                   decay=decay[i],
                   min_dist=min_dist[i],
                   max_dist=max_dist[i],
                   distance_curve=distance_curve[i],
                   min_freq=min_freq[i],
                   max_freq=max_freq[i],
                   frequency_curve=frequency_curve[i],
                   H=H,
                   prime_list=PRIME_LIST,
                   sr=SAMPLE_RATE)
    impulse_responses[i, :] = y

    # Calculate the MFCCs of the processed audio
    mfccs = mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=16, n_fft = 1024, hop_length=256)
    # print('before reshape', mfccs.shape)
    
    # reshaped mfcc.
    mfccs = mfccs.reshape(mfccs.shape[:-2] + (-1,))
    # print('after reshape', mfccs.shape)

    MFCCS.append(mfccs)

    # # Save MFCCs to a .txt file
    mfcc_path = mfcc_dir + '/' + "mfcc" + f"_{i}.txt"
    np.savetxt(mfcc_path, mfccs, fmt='%.10f')

    # Write impulse responses to y folder
    y_path = y_dir + '/' + "impulse" + f"_{i}.wav"
    write(y_path, y, SAMPLE_RATE)
    
    # print(f"Generated: {i+1}/{IMPULSE_NUM}")

# convert MFCCS to numpy array
MFCCS = np.array(MFCCS)
print(f"{IMPULSE_NUM} samples generated.")
# print(f"{P.shape} parameters calculated.")
print(f"{MFCCS.shape} MFCCS calculated.")

print("shape checking MFCCS: ", MFCCS.shape)

# Normalise each array of MFCCs
for i in range(MFCCS.shape[0]):
    MFCCS[i] = (MFCCS[i] - MFCCS[i].min()) / (MFCCS[i].max() - MFCCS[i].min())

# Store the normalised MFCCs in a .txt file
for i in range(IMPULSE_NUM):
    mfcc_n_path = mfcc_n_dir + '/' + "mfcc" + f"_{i}.txt" 
    np.savetxt(mfcc_n_path, MFCCS[i], fmt='%.10f')

# Save parameters.
np.savetxt(r'data\fdn_dataset\p\p.txt', parameters.values, fmt='%10.4f')

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")