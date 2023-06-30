import numpy as np
import torch


@torch.jit.script
def allpass(input_signal: torch.Tensor, delay: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """
    Apply a differentiable allpass filter to the input signal.

    Parameters:
        input_signal (torch.Tensor): Input signal to be filtered.
        delay (int): Delay length in samples.

    Returns:
        torch.Tensor: Filtered output signal.

    """
    delay = torch.clamp(delay, 1, 44100)
    gain = torch.clamp(gain, 0, 0.999)
    # Create delay line buffer
    delay_line = torch.zeros(delay.item(), device=input_signal.device)
    output_signal = torch.zeros_like(input_signal)

    for i, x in enumerate(input_signal):
        delayed_sample = delay_line[-1]
        output_signal[i] = -gain * x + delayed_sample + gain * delay_line[0]
        delay_line = torch.roll(delay_line, shifts=-1, dims=0)
        delay_line[-1] = x

    return output_signal

@torch.jit.script
def comb(input_signal: torch.Tensor, delay: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """
    Apply a differentiable comb filter to the input signal.

    Parameters:
        input_signal (torch.Tensor): Input signal to be filtered.
        delay (int): Delay length in samples.
        gain (float): Gain coefficient for the comb filter.
        feedback (bool): Feedback mode, feedforward otherwise.
    Returns:
        torch.Tensor: Filtered output signal.

    """
    delay = torch.clamp(delay, 1, 44100)
    gain = torch.clamp(gain, 0, 0.999)
    # Create delay line buffer
    delay_line = torch.zeros(delay.item(), device=input_signal.device)
    output_signal = torch.zeros_like(input_signal)

    for i, x in enumerate(input_signal):
        delayed_sample = delay_line[-1]
        output_signal[i] = x + gain * delayed_sample
        delay_line = torch.roll(delay_line, shifts=-1, dims=0)
        delay_line[0] = output_signal[i]  # Feed back output into the delay line

    return output_signal


@torch.jit.script
def freeverb(input_signal: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Apply the differentiable Freeverb algorithm to the input signal.

    Parameters:
        input_signal (torch.Tensor): Input signal to be processed.
    Returns:
        torch.Tensor: Processed output signal.

    """

    # Set default values for the parameters if they are not provided
    c_params = params[:16]
    c_delays = c_params[:8]
    c_gains = c_params[8:]
    a_params = params[16:]
    a_delays = a_params[:4] 
    a_gains = a_params[4:]

    if c_delays is None:
        c_delays = torch.tensor([1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116])
    if c_gains is None:
        c_gains = torch.tensor([.84, .84, .84, .84, .84, .84, .84, .84])
    if a_delays is None:
        a_delays = torch.tensor([225, 556, 441, 341])
    if a_gains is None:
        a_gains = torch.tensor([0.5, 0.5, 0.5, 0.5])

    # Apply comb filters
    output_signal = torch.zeros_like(input_signal)
    for delay, gain in zip(c_delays, c_gains):
        c = comb(input_signal, delay, gain)
        output_signal = torch.add(
            output_signal,
            c
        )

    # Apply allpass filters
    for delay, gain in zip(a_delays, a_gains):
        output_signal = allpass(output_signal, delay, gain)

    return output_signal
