import numba
import numpy as np
from librosa import stft, power_to_db, display
from matplotlib import pyplot as plt
import os

@numba.jit(nopython=True)
def pad(a, pad_width, constant_values=0):
    """
    Pads a 1D NumPy array with a constant value.

    Parameters:
    a (ndarray): Input array.
    pad_width (int or sequence of ints): Number of values padded to the edges of each axis.
    constant_values (scalar or sequence): Values used for padding. Default is 0.

    Returns:
    ndarray: Padded array.

    Note:
    This function works only for 1D arrays.
    """
    a_ = np.zeros(a.shape[-1] + pad_width)
    a_[:a.shape[-1]] = a
    return a_

@numba.jit(nopython=True)
def padadd(a, b):
    """
    Adds two 1D NumPy arrays, padding the shorter array if their lengths do not match.

    Parameters:
    a (ndarray): First input array.
    b (ndarray): Second input array.

    Returns:
    ndarray: Sum of the two arrays with padding.

    Note:
    This function works only for 1D arrays.
    """
    len_a = a.shape[-1]
    len_b = b.shape[-1]
    max_len = max(len_a, len_b)
    if len_a < max_len:
        a = pad(a, max_len - len_a)
    if len_b < max_len:
        b = pad(b, max_len - len_b)
    return a + b

###
# Util functions from NeuralReverberator repo by C. Steinmetz
###
def generate_specgram(x, sr=16000, n_fft=1024, n_hop=256):
    """
    Generate a spectrogram (via stft) on input audio data.

    Args:
        x (ndarray): Input audio data.
        sr (int, optional): Sample rate out input audio data.
        n_fft (int, optional): Size of the FFT to generate spectrograms.
        n_hop (int, optional): Hop size for FFT.
    """
    S = stft(x, n_fft=n_fft, hop_length=n_hop, center=True)
    power_spectra = np.abs(S)**2
    log_power_spectra = power_to_db(power_spectra)
    _min = np.amin(log_power_spectra)
    _max = np.amax(log_power_spectra)
    if _min == _max:
        print(f"divide by zero in audio array")
    else:
        normalized_log_power_spectra = (log_power_spectra - _min) / (_max - _min)
    return normalized_log_power_spectra
    

def plot_specgram(log_power_spectra, rate, filename, output_dir):
    """ 
    Save log-power and normalized log-power specotrgram to file.

    Args:
        log_power_spectra (ndarray): Comptued Log-Power spectra.
        rate (int): Sample rate of input audio data.
        filename (str): Output filename for generated plot.
        output_dir (str): Directory to save generated plot.
    """

    plt.figure()
    display.specshow(log_power_spectra, sr=rate*2, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Power spectrogram')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close('all')