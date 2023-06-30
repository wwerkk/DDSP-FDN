import numpy as np
from dsp import iir
def freeverb(input_signal, sample_rate, gain, delay, decay):
    """
    Apply the Freeverb algorithm to the input signal.

    Parameters:
        input_signal (array_like):
            Input signal to be processed.
        sample_rate (int):
            Sample rate of input signal.
        gain (float):
            Gain coefficient for the input signal.
        delay (float):
            Delay time for the comb filters in seconds.
        decay (float):
            Decay factor for the comb filters.

    Returns:
        ndarray:
            Processed output signal.

    Example:
        import numpy as np

        # Generate an input signal
        sample_rate = 44100
        duration = 5.0
        t = np.linspace(0.0, duration, int(sample_rate * duration))
        input_signal = np.sin(2.0 * np.pi * 440.0 * t)

        # Apply Freeverb to the input signal
        gain = 0.5
        delay = 0.03
        decay = 0.8
        output_signal = freeverb(input_signal, gain, delay, decay)
    """
    # Convert delay time to number of samples
    delay_samples = int(delay * sample_rate)

    # Define the filter coefficients
    comb_feedback = np.array([decay])
    comb_feedforward = np.array([1.0])
    comb_A = np.diag(comb_feedback)
    comb_B = np.zeros((1, 1))
    comb_B[0, 0] = gain
    comb_C = np.zeros((1, 1))
    comb_C[0, 0] = 1.0
    comb_D = np.zeros((1, 1))

    allpass_feedback = np.array([0.5])
    allpass_feedforward = np.array([0.5])
    allpass_A = np.diag(allpass_feedback)
    allpass_B = np.zeros((1, 1))
    allpass_C = np.zeros((1, 1))
    allpass_C[0, 0] = 1.0
    allpass_D = np.zeros((1, 1))

    # Apply the comb filters
    comb_output = iir(input_signal, comb_A, comb_B, comb_C, comb_D)

    # Apply the allpass filters
    allpass_output = iir(comb_output, allpass_A, allpass_B, allpass_C, allpass_D)

    # Combine the original input signal with the processed signal
    output_signal = input_signal + allpass_output

    return output_signal
