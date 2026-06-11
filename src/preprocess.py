"""Signal preprocessing: zero-phase bandpass filter and common avg reference."""
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(x, fs=100, low=1.0, high=45.0, order=4):
    """Zero-phase Butterworth bandpass along the last axis.

    x : ndarray (..., n_samples)
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=float), axis=-1)


def common_average_reference(epoch):
    """Subtract the mean across channels at each time point.

    epoch : ndarray (n_channels, n_samples)
    """
    epoch = np.asarray(epoch, dtype=float)
    return epoch - epoch.mean(axis=0, keepdims=True)
