import numpy as np
import numpy.typing as npt
from numba import njit
from util import pad

@njit
def comb(x: npt.NDArray[np.float64], b: float = 1.0, M: int = 2000, a: float = 0.9) -> np.ndarray[np.float64]:
    """
    Implements a feedback comb filter.

    Args:
        x (np.ndarray): Input signal.
        b (float, optional): Input gain. Defaults to 1.0.
        M (int, optional): Delay time in samples. Defaults to 2000.
        a (float, optional): Feedback gain. Defaults to 0.9.

    Returns:
        np.ndarray: Output signal.
    """
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in range(y.shape[-1]):
        if i < x.shape[-1]:
            y[i] += b * x[i]
        if i >= M:
            y[i] += feedback
            feedback = -a * y[i - M]
    return y

@njit
def lbcf(x: npt.NDArray[np.float64], b: float = 1.0, M: int = 2000, a: float = 0.9, d: float = 0.5) -> np.ndarray[np.float64]:
    """
    Implements Schroeder's Lowpass-Feedback Comb Filter.

    Args:
        x (np.ndarray): Input signal.
        b (float, optional): Input gain. Defaults to 1.0.
        M (int, optional): Delay time in samples. Defaults to 2000.
        a (float, optional): Feedback gain. Defaults to 0.9.
        d (float, optional): Damping factor. Defaults to 0.5.

    Returns:
        np.ndarray: Output signal.
    """
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in range(y.shape[-1]):
        if i < x.shape[-1]:
            y[i] += b * x[i]
        if i >= M:
            y[i] += feedback
            feedback += (1 - d) * ((a * y[i - M]) - feedback)
    return y

@njit
def allpass(x: npt.NDArray[np.float64], M: int = 2000, a: float = 0.5) -> npt.NDArray[np.float64]:
    """
    Implements an allpass filter.

    Args:
        x (np.float32): Input signal.
        M (int, optional): Delay time in samples. Defaults to 2000.
        a (float, optional): Feedback gain. Defaults to 0.5.

    Returns:
        np.float32: Output signal.
    """
    feedback = 0
    y = np.zeros(x.shape[-1] + M)
    feedback = 0
    for i in range(y.shape[-1]):
        if i < x.shape[-1]:
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
    """
    Applies Freeverb algorithm to the input signal.

    Args:
        x (np.ndarray): Input signal.
        cb (optional): List of input gains for parallel lowpass-feedback comb filters. Defaults to [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].
        cM (optional): List of delay times in samples for parallel lowpass-feedback comb filters. Defaults to [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116].
        ca (optional): List of feedback gains for parallel lowpass-feedback comb filters. Defaults to [0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84].
        cd (optional): List of damping factors for parallel lowpass-feedback comb filters. Defaults to [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2].
        aM (optional): List of delay times in samples for cascading allpass filters. Defaults to [225, 556, 441, 341].
        aa (optional): List of feedback gains for cascading allpass filters. Defaults to [0.5, 0.5, 0.5, 0.5].

    Returns:
        np.ndarray: Output signal.
    """
    y = np.zeros(x.shape[-1])
    # Apply paralell low-passed feedback comb filters
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

    max_abs_value = np.max(np.abs(y))
    epsilon = 1e-12
    y = y / (max_abs_value + epsilon)
    return y