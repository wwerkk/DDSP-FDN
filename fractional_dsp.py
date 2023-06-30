
def fractional_delay(input_signal, delay):
    """
    Apply a fractional delay to the input signal.

    Parameters:
        input_signal (array_like): Input signal to be delayed.
        delay (float): Delay amount in fractional samples.

    Returns:
        ndarray: Delayed output signal.
    """
    output_signal = np.zeros_like(input_signal)
    for i in range(len(input_signal)):
        index = i - delay
        index_floor = int(np.floor(index))
        frac = index - index_floor
        if index_floor < 0 or index_floor >= len(input_signal) - 1:
            continue
        output_signal[i] = (1 - frac) * input_signal[index_floor] + frac * input_signal[index_floor + 1]
    return output_signal


def fractional_allpass(input_signal, delay, gain):
    """
    Apply an allpass filter with fractional delay to the input signal.

    Parameters:
        input_signal (array_like): Input signal to be filtered.
        delay (float): Delay amount in fractional samples.
        gain (float): Gain coefficient for the allpass filter.

    Returns:
        ndarray: Filtered output signal.
    """
    delayed_signal = fractional_delay(input_signal, delay)
    return -gain * input_signal + delayed_signal + gain * delayed_signal[0]


def fractional_comb(input_signal, delay, gain):
    """
    Apply a comb filter with fractional delay to the input signal.

    Parameters:
        input_signal (array_like): Input signal to be filtered.
        delay (float): Delay amount in fractional samples.
        gain (float): Gain coefficient for the comb filter.

    Returns:
        ndarray: Filtered output signal.
    """
    delayed_signal = fractional_delay(input_signal, delay)
    return input_signal + gain * delayed_signal



def fractional_freeverb(input_signal, fs, rt60):
    """
    Apply the Freeverb algorithm with fractional delay to the input signal.

    Parameters:
        input_signal (array_like): Input signal to be processed.
        fs (int): Sample rate of the input signal.
        rt60 (float): Reverberation time in seconds for a decay of 60dB.

    Returns:
        ndarray: Processed output signal.
    """
    # Define delay lengths
    allpass_delays = [347.7, 113.1, 37.8]  # in samples
    comb_delays = [1687.3, 1601.5, 2053.7, 2251.1]  # in samples

    # Calculate decay factors for 60dB decay
    allpass_gains = [0.7, 0.7, 0.7]
    comb_gains = [10 ** (-3 * d / (fs * rt60)) for d in comb_delays]

    # Apply allpass filters
    for delay, gain in zip(allpass_delays, allpass_gains):
        input_signal = allpass(input_signal, delay, gain)

    # Apply comb filters
    output_signal = np.zeros_like(input_signal)
    for delay, gain in zip(comb_delays, comb_gains):
        output_signal += comb(input_signal, delay, gain)

    return output_signal