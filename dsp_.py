import numpy as np
from typing import Optional
import numba

# Process audio using numba (a bit faster)

@numba.jit(nopython=True)
def allpass(x, M=2000, a=0.5):
    feedback = 0
    y = np.zeros(x.shape[-1] + M * 4)
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

@numba.jit(nopython=True)
def comb(x, b=1, M=11050, a=0.5):
    y = np.zeros(x.shape[-1] + M * 4)
    feedback = 0
    for i in np.arange(0, len(y)):
        if i < (x.shape[-1]):
            y[i] += b * x[i]
        if i >= M:
            y[i] += feedback
            feedback = y[i - M]
            feedback *= -a
    return y

# Schroeder's Lowpass-Feedback Comb Filter
# https://ccrma.stanford.edu/~jos/pasp/Lowpass_Feedback_Comb_Filter.html
@numba.jit(nopython=True)
def lbcf(x, b=1, M=11050, a=0.5, d=0.5):
    y = np.zeros(x.shape[-1] + M * 4)
    feedback = 0
    for i in np.arange(0, len(y)):
        if i < (x.shape[-1]):
            y[i] += b * x[i]
        if i >= M:
            y[i] += feedback
            feedback += d * ((y[i - M]) - feedback)
            feedback *= -a
    return y


def freeverb(input_signal: np.ndarray, c_delays: Optional[np.ndarray] = None, c_gains: Optional[np.ndarray] = None, c_damps: Optional[np.ndarray] = None, a_delays: Optional[np.ndarray] = None, a_gains: Optional[np.ndarray] = None) -> np.ndarray:
    if c_delays is None:
        c_delays = np.array([1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116])
    if c_gains is None:
        c_gains = np.array([0.84 for i in range(8)])
    if c_damps is None:
        c_damps = np.array([0.2 for i in range(8)])
    if a_delays is None:
        a_delays = np.array([225, 556, 441, 341])
    if a_gains is None:
        a_gains = np.array([0.5 for i in range(4)])

    # input_signal *= 0.015 # scale input

    # Apply comb filters
    output_signal = np.zeros_like(input_signal)
    for delay, gain, damp in zip(c_delays, c_gains, c_damps):
        lbcf_out = lbcf(
            x=input_signal,
            M=delay,
            d=damp,
            a=gain
            )
        shape = output_signal.shape[-1]
        shape_ = lbcf_out.shape[-1]
        if shape < shape_:
            # print(shape, shape_, shape_-shape)
            output_signal = np.pad(
                output_signal,
                (0, shape_-shape), 'constant', constant_values=(0))
        elif shape > shape_:
            # print(shape, shape_, shape-shape_)
            lbcf_out = np.pad(
                lbcf_out,
                (0, shape-shape_), 'constant', constant_values=(0))
        output_signal += lbcf_out
        
    # Apply allpass filters
    for delay, gain in zip(a_delays, a_gains):
        output_signal = allpass(output_signal, delay, gain)

    # Normalize output
    # max_abs_value = np.max(np.abs(output_signal))
    # epsilon = 1e-12
    # output_signal = output_signal / (max_abs_value + epsilon)
    return output_signal