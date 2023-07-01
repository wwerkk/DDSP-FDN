from iir import iir

import numpy as np
import soundfile as sf

# Load the input audio signal
audio_data, sr = sf.read('data/iir_dataset/dry/balloon_burst_1.wav')

# Normalize the audio signal to the range [-1, 1]
# audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
# audio_data = audio_data.reshape(1, -1)

# Define the filter coefficients with compatible data types
A = np.array([[1.0, -0.8, 0.2],
              [0.0, 0.5, -0.5],
              [0.0, 0.2, 0.9]], dtype=np.float32)
B = np.array([[1.0],
              [0.0],
              [0.0]], dtype=np.float32)
C = np.array([[0.5, 0.6, 0.8]], dtype=np.float32)
D = np.array([[0.6]], dtype=np.float32)

# print(audio_data.shape)
# print(audio_data.dtype)
# print(A.shape)
# print(A.dtype)
# print(B.shape)
# print(B.dtype)
# print(C.shape)
# print(D.shape)

# Apply the IIR filter to the audio signal
# lfilter(B, A, audio_data)
filtered_audio = iir(audio_data, A, B, C, D)
print(filtered_audio.shape)

# Save the filtered audio to a file
# filtered_audio = (filtered_audio * np.iinfo(audio_data.dtype).max).astype(audio_data.dtype)
sf.write('data/iir_dataset/wet/balloon_burst_1.wav', filtered_audio, sr)