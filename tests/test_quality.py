import numpy as np
from src.quality import (check_sampling_regularity, flat_channels,
                         epoch_is_artifact)

def test_check_sampling_regularity_ok():
    ts = np.arange(0, 10, 0.01)   # perfect 100 Hz
    res = check_sampling_regularity(ts, fs=100, tol=0.2)
    assert res["regular"] is True
    assert abs(res["median_fs"] - 100) < 1

def test_check_sampling_regularity_gap():
    ts = np.concatenate([np.arange(0, 1, 0.01), np.arange(5, 6, 0.01)])
    res = check_sampling_regularity(ts, fs=100, tol=0.2)
    assert res["regular"] is False

def test_flat_channels_detects_constant():
    sig = np.random.randn(1000, 3)
    sig[:, 1] = 7.0   # constant channel
    flats = flat_channels(sig, flat_std=0.5)
    assert flats == [1]

def test_epoch_is_artifact_high_ptp():
    good = np.random.randn(16, 300) * 5      # small amplitude
    bad = good.copy()
    bad[0, 0] = 5000                          # huge spike -> high ptp
    assert epoch_is_artifact(good, ptp_thresh=500, flat_std=0.5) is False
    assert epoch_is_artifact(bad, ptp_thresh=500, flat_std=0.5) is True

def test_epoch_is_artifact_flat():
    flat = np.ones((16, 300)) * 3.0
    assert epoch_is_artifact(flat, ptp_thresh=500, flat_std=0.5) is True
