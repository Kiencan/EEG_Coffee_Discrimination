import numpy as np
from src.features import band_powers, feature_names
from src.config import BANDS, EEG_CHANNELS

def test_feature_vector_length():
    epoch = np.random.randn(16, 300)
    feats = band_powers(epoch, fs=100)
    assert feats.shape == (2 * len(BANDS) * 16,)

def test_feature_names_match_length():
    names = feature_names()
    assert len(names) == 2 * len(BANDS) * len(EEG_CHANNELS)
    assert "abs_alpha_O1" in names and "rel_alpha_O1" in names

def test_alpha_sine_dominates_alpha_band():
    fs = 100
    t = np.arange(0, 3, 1/fs)
    epoch = np.random.randn(16, len(t)) * 0.01
    epoch[0] = np.sin(2*np.pi*10*t)
    feats = band_powers(epoch, fs=fs)
    n = len(BANDS) * 16
    band_list = list(BANDS.keys())
    alpha_i = band_list.index("alpha")
    rel = feats[n:].reshape(len(BANDS), 16)
    assert np.argmax(rel[:, 0]) == alpha_i
