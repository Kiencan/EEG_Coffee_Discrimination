import numpy as np
from src.preprocess import bandpass_filter, common_average_reference

def test_bandpass_removes_dc_offset():
    fs = 100
    t = np.arange(0, 3, 1/fs)
    # 10 Hz sine + large DC offset, shape (1 channel, n_samples)
    x = (50 + np.sin(2*np.pi*10*t))[None, :]
    y = bandpass_filter(x, fs=fs, low=1, high=45, order=4)
    assert abs(np.mean(y)) < 1.0           # DC removed
    assert np.std(y) > 0.3                  # 10 Hz component preserved

def test_bandpass_attenuates_high_freq():
    fs = 100
    t = np.arange(0, 3, 1/fs)
    low10 = np.sin(2*np.pi*10*t)[None, :]
    high48 = np.sin(2*np.pi*48*t)[None, :]
    y_low = bandpass_filter(low10, fs=fs, low=1, high=45)
    y_high = bandpass_filter(high48, fs=fs, low=1, high=45)
    assert np.std(y_high) < np.std(y_low)   # 48 Hz attenuated vs 10 Hz

def test_common_average_reference_zero_sum():
    epoch = np.random.randn(16, 300)
    car = common_average_reference(epoch)
    assert np.allclose(car.mean(axis=0), 0, atol=1e-10)
