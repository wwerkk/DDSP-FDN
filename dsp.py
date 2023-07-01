import numpy as np
from typing import Optional
import numba
# Process audio using numba (a bit faster)

@numba.jit(nopython=True)
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

# Schroeder's Lowpass-Feedback Comb Filter
# https://ccrma.stanford.edu/~jos/pasp/Lowpass_Feedback_Comb_Filter.html
@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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
    # Apply parallel lowpass feedback comb filters
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
        
    # Apply cascading allpass filters
    for M, a in zip(aM, aa):
        y = allpass(y, M, a)

    # Normalize output
    max_abs_value = np.max(np.abs(y))
    epsilon = 1e-12
    y = y / (max_abs_value + epsilon)
    return y