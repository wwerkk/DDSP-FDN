import numpy as np
import soundfile as sf

# TODO: Something's still off with the feedback!

def comb(x, b=1.0, M=2000, a=0.9):
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in np.arange(0, len(y)):
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
            feedback += d * ((a * y[i - M]) - feedback)
    return y

def allpass(x, M=2000, a=0.5):
    feedback = 0
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in np.arange(0, len(y)):
        if i < (x.shape[-1]):
            y[i] = x[i] - feedback
            feedback *= a
            feedback += x[i]
        else:
            y[i] -= feedback
            feedback *= a
    return y



x = np.array([0, 1.0, 1.0, 1.0], dtype=np.float32)

x = comb(x, 0.5, 2, 1.0)
print(x)



x, sr = sf.read('/Users/wwerkowicz/GitHub/DDSP-FDN/data/comb_dataset/input/dry/balloon_burst_1.wav')

# x = comb(x, 1.0, 11050, 0.5)
x = lbcf(x, 1.0, 11050, 0.5, 0.5)

sf.write('balloon_burst_comb.wav', x, sr)