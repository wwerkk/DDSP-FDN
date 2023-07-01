import numpy as np
import soundfile as sf
import os

def comb(x, b=1.0, M=2000, a=0.9):
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in range(len(y)):
        if i < (x.shape[-1]):
            y[i] += b * x[i]
        if i >= M:
            y[i] += feedback
            feedback = -a * y[i - M]
    return y

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

def freeverb(
        x,
        cb=[1.0 for i in range(8)],
        cM=[1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116],
        ca=[0.84 for i in range(8)],
        cd=[0.2 for i in range(8)],
        aM=[225, 556, 441, 341],
        aa=[0.5 for i in range(4)]
        ):
    # Apply comb filters
    y = np.zeros_like(x)
    for b, M, a, d in zip(cb, cM, ca, cd):
        y_ = lbcf(
            x=x,
            b=b,
            M=M,
            a=a,
            d=d
            )
        shape = y.shape[-1]
        shape_ = y_.shape[-1]
        if shape < shape_:
            # print(shape, shape_, shape_-shape)
            y = np.pad(
                y,
                (0, shape_-shape), 'constant', constant_values=(0))
        elif shape > shape_:
            # print(shape, shape_, shape-shape_)
            y_ = np.pad(
                y_,
                (0, shape-shape_), 'constant', constant_values=(0))
        y += y_
        
    # Apply allpass filters
    for M, a in zip(aM, aa):
        y = allpass(y, M, a)

    # Normalize output
    max_abs_value = np.max(np.abs(y))
    epsilon = 1e-12
    y = y / (max_abs_value + epsilon)
    return y

# x = np.array([0, 1.0, 1.0, 1.0], dtype=np.float32)

# x = comb(x, 0.5, 2, 1.0)
# print(x)

file = 'data/test/noise_burst.wav'
x, sr = sf.read(file)

y = comb(x, 0.5, 8000, 0.7)
file_ = os.path.splitext(file)[0] + f"_comb.wav"
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