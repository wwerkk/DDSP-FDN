import numpy as np
from typing import List, Optional

def allpass(input_signal, delay, gain):
    """
    Apply an allpass filter to the input signal.

    Parameters:
        input_signal (array_like): Input signal to be filtered.
        delay (int): Delay length in samples.
        gain (float): Gain coefficient for the allpass filter.

    Returns:
        ndarray: Filtered output signal.
    """
    delay = np.clip(delay, 20, 20000)
    gain = np.clip(gain, 0.0, 0.99)
    delay_line = np.zeros(delay)
    output_signal = np.zeros_like(input_signal)
    for i, x in enumerate(input_signal):
        output_signal[i] = -gain * x + delay_line[-1] + gain * delay_line[0]
        delay_line = np.roll(delay_line, -1)
        delay_line[-1] = x
    return output_signal


def comb(input_signal, delay, gain):
    """
    Apply a comb filter to the input signal.

    Parameters:
        input_signal (array_like): Input signal to be filtered.
        delay (int): Delay length in samples.
        gain (float): Gain coefficient for the comb filter.

    Returns:
        ndarray: Filtered output signal.
    """
    delay = np.clip(delay, 20, 20000)
    gain = np.clip(gain, 0.0, 0.99)
    delay_line = np.zeros(delay)
    output_signal = np.zeros_like(input_signal)
    for i, x in enumerate(input_signal):
        output_signal[i] = x + gain * delay_line[-1]
        delay_line = np.roll(delay_line, -1)
        delay_line[-1] = output_signal[i]
    return output_signal


def freeverb(input_signal: np.ndarray, c_delays: Optional[List[int]] = None, c_gains: Optional[List[float]] = None, a_delays: Optional[List[int]] = None, a_gains: Optional[List[float]] = None) -> np.ndarray:
    """
    Apply the Freeverb (aka Schroeder Reverb) algorithm to the input signal.

    Parameters:
        input_signal (array_like): Input signal to be processed.
        fs (int): Sample rate of the input signal.
        rt60 (float): Reverberation time in seconds for a decay of 60dB.

    Returns:
        ndarray: Processed output signal.
    """

    # Set default values for the parameters if they are not provided
    if c_delays is None:
        c_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
    if c_gains is None:
        c_gains = [.84, .84, .84, .84, .84, .84, .84, .84]
    if a_delays is None:
        a_delays = [225, 556, 441, 341]
    if a_gains is None:
        a_gains = [0.5, 0.5, 0.5, 0.5]

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


def ssr_iir(input_signal, A, B, C, D):
    """
    Apply an Infinite Impulse Response (IIR) filter using state space realization.

    Parameters:
        input_signal (array_like): Input signal to be filtered.
        A (ndarray): State transition matrix of shape (num_states, num_states).
        B (ndarray): Input matrix of shape (num_states, 1).
        C (ndarray): Output matrix of shape (1, num_states).
        D (ndarray): Feedforward matrix of shape (1, 1).

    Returns:
        ndarray: Filtered output signal.

    Example:
        import numpy as np
        from scipy.io import wavfile

        # Load the input audio signal
        sample_rate, audio_data = wavfile.read('input_audio.wav')

        # Normalize the audio signal to the range [-1, 1]
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

        # Define the filter coefficients
         >>> A = np.array([[1.0, -0.8, 0.2],
                      [0.0, 0.5, -0.5],
                      [0.0, 0.2, 0.9]])
        >>> B = np.array([[1.0],
                      [0.0],
                      [0.0]])
        >>> C = np.array([[0.5, 0.3, 0.1]])
        >>> D = np.array([[0.2]])

        # Apply the IIR filter to the audio signal
        >>> filtered_audio = iir(audio_data, A, B, C, D)

        # Save the filtered audio to a file
        >>> filtered_audio = (filtered_audio * np.iinfo(audio_data.dtype).max).astype(audio_data.dtype)
        >>> wavfile.write('output_audio.wav', sample_rate, filtered_audio)
    """
    # Initialize the output signal and state vector
    num_samples = len(input_signal)
    num_states = A.shape[0]
    output_signal = np.zeros(num_samples)
    x = np.zeros(num_states)
    # Apply the state space equations to each input sample
    for i in range(num_samples):
        x = np.dot(A, x) + np.dot(B, input_signal[i])
        # Take mean of outputs for single channel out
        output_signal[i] = np.mean(np.dot(C, x)) + np.mean(np.dot(D, input_signal[i]))
    return output_signal