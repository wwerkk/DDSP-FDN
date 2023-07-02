import numpy as np
import soundfile as sf
import os
from util import pad
from numba import njit

@njit
def fbcf(x, b=1.0, M=2000, a=0.9):
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in range(len(y)):
        if i < (x.shape[-1]):
            y[i] += b * x[i]
        if i >= M:
            y[i] += feedback
            feedback = -a * y[i - M]
    return y

@njit
def lbcf(x, b=1.0, M=2000, a=0.9, d=0.5):
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in np.arange(0, len(y)):
        if i < (x.shape[-1]):
            y[i] += b * x[i]
        if i >= M:
            y[i] += feedback
            feedback += (1 - d) * ((a * y[i - M]) - feedback)
    return y

@njit
def allpass(x, M=2000, a=0.5):
    feedback = 0
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in np.arange(0, len(y)):
        if i < (x.shape[-1]):
            y[i] = x[i] - feedback
            feedback *= a
            if i >= M:
                feedback += x[i]
        else:
            y[i] -= feedback
            feedback *= a
    return y

@njit
def freeverb(
        x,
        cb=np.full(8, 1.0),
        cM=np.array([1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116], dtype=np.int64),
        ca=np.full(8, 0.84),
        cd=np.full(8, 0.2),
        aM=np.array([225, 556, 441, 341], dtype=np.int64),
        aa=np.full(4, 0.5)
        ):
    # Apply paralell low-passed feedback comb filters
    y = np.zeros_like(x)
    for b, M, a, d in zip(cb, cM, ca, cd):
        y_ = lbcf(x=x, b=b, M=M, a=a, d=d)
        shape = y.shape[-1]
        shape_ = y_.shape[-1]
        if shape < shape_:
            y = pad(y, shape_-shape)
        elif shape > shape_:
            y_ = pad(y_, shape-shape_)
        y += y_
    # Apply cascading allpass filters
    for M, a in zip(aM, aa):
        y = allpass(y, M, a)
    return y


# TODO: Jot's reverberator
# https://ccrma.stanford.edu/~jos/pasp/History_FDNs_Artificial_Reverberation.html
# https://ccrma.stanford.edu/~jos/pasp/img745_2x.png

# x = np.array([0, 1.0, 1.0, 1.0], dtype=np.float32)

# x = fbcf(x, 0.5, 2, 1.0)
# print(x)

file = 'data/test/noise_burst.wav'
x, sr = sf.read(file)

y = fbcf(x, 0.5, 8000, 0.7)
file_ = os.path.splitext(file)[0] + f"_fbcf.wav"
sf.write(file_, y, sr)
y = lbcf(x, 0.5, 8000, 0.7, 0.7)
file_ = os.path.splitext(file)[0] + f"_lbcf.wav"
sf.write(file_, y, sr)
y = allpass(x, 2000, 0.7)
file_ = os.path.splitext(file)[0] + f"_allpass.wav"
sf.write(file_, y, sr)
y = freeverb(x=x)
file_ = os.path.splitext(file)[0] + f"_freeverb.wav"
sf.write(file_, y, sr)