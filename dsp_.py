import numpy as np
from typing import List, Optional
import numba

# Process audio using numba (a bit faster)

@numba.jit(nopython=True)
def allpass(input_signal, delay, gain):
    delay_line = np.zeros(delay)
    output_signal = np.zeros_like(input_signal)
    for i, x in enumerate(input_signal):
        output_signal[i] = -gain * x + delay_line[-1] + gain * delay_line[0]
        delay_line = np.roll(delay_line, -1)
        delay_line[-1] = x
    return output_signal

@numba.jit(nopython=True)
def comb(input_signal, delay, gain):
    delay_line = np.zeros(delay)
    output_signal = np.zeros_like(input_signal)
    for i, x in enumerate(input_signal):
        output_signal[i] = x + gain * delay_line[-1]
        delay_line = np.roll(delay_line, -1)
        delay_line[-1] = output_signal[i]
    return output_signal


def freeverb(input_signal: np.ndarray, c_delays: Optional[np.ndarray] = None, c_gains: Optional[np.ndarray] = None, a_delays: Optional[np.ndarray] = None, a_gains: Optional[np.ndarray] = None) -> np.ndarray:
    if c_delays is None:
        c_delays = np.array([1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116])
    if c_gains is None:
        c_gains = np.array([.84, .84, .84, .84, .84, .84, .84, .84])
    if a_delays is None:
        a_delays = np.array([225, 556, 441, 341])
    if a_gains is None:
        a_gains = np.array([0.5, 0.5, 0.5, 0.5])

    # Apply allpass filters
    for delay, gain in zip(a_delays, a_gains):
        input_signal = allpass(input_signal, delay, gain)

    # Apply comb filters
    output_signal = np.zeros_like(input_signal)
    for delay, gain in zip(c_delays, c_gains):
        output_signal += comb(input_signal, delay, gain)
    # Normalize output
    max_abs_value = np.max(np.abs(output_signal))
    epsilon = 1e-12
    output_signal = output_signal / (max_abs_value + epsilon)
    return output_signal