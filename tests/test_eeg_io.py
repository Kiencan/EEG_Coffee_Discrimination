import numpy as np
from src.eeg_io import find_code_runs, label_for_code

def test_find_code_runs_basic():
    codes = np.array([0, 0, 1, 1, 1, 0, 981, 981, 981])
    runs = find_code_runs(codes)
    assert runs == [(0, 0, 2), (1, 2, 3), (0, 5, 1), (981, 6, 3)]

def test_find_code_runs_empty():
    assert find_code_runs(np.array([])) == []

def test_label_for_code():
    assert label_for_code(981) == "Arabica"
    assert label_for_code(585) == "Robusta"
    assert label_for_code(712) is None   # control
    assert label_for_code(0) is None

from src.eeg_io import extract_epochs

def test_extract_epochs_one_per_run():
    from src.config import EPOCH_LEN
    n_ch = 16
    rest = np.zeros((5, n_ch))
    ara = np.ones((EPOCH_LEN, n_ch)) * 1.0
    rob = np.ones((EPOCH_LEN, n_ch)) * 2.0
    signals = np.vstack([rest, ara, rest, rob, rest])
    codes = np.concatenate([
        np.zeros(5), np.full(EPOCH_LEN, 981), np.zeros(5),
        np.full(EPOCH_LEN, 585), np.zeros(5)])
    X, y, run_codes = extract_epochs(signals, codes, expected_len=EPOCH_LEN)
    assert X.shape == (2, n_ch, EPOCH_LEN)
    assert list(y) == ["Arabica", "Robusta"]
    assert list(run_codes) == [981, 585]
    assert np.allclose(X[0], 1.0) and np.allclose(X[1], 2.0)

def test_extract_epochs_skips_short_and_control():
    from src.config import EPOCH_LEN
    n_ch = 16
    short = np.ones((EPOCH_LEN - 50, n_ch))
    ctrl = np.ones((EPOCH_LEN, n_ch))
    signals = np.vstack([short, ctrl])
    codes = np.concatenate([np.full(EPOCH_LEN - 50, 981),
                            np.full(EPOCH_LEN, 712)])
    X, y, _ = extract_epochs(signals, codes, expected_len=EPOCH_LEN)
    assert X.shape[0] == 0

def test_load_subject_drops_empty_rows(tmp_path):
    import pandas as pd
    from src.config import EEG_CHANNELS
    from src.eeg_io import load_subject
    n = 5
    data = {"times": list(range(n + 2)),
            "timestamp": [0.01 * i for i in range(n + 2)]}
    for ch in EEG_CHANNELS:
        data[ch] = list(range(n)) + [np.nan, np.nan]   # 2 trailing empty rows
    data["ECG1"] = [0] * (n + 2)
    data["ECG2"] = [0] * (n + 2)
    data["code"] = [0] * n + [np.nan, np.nan]
    df = pd.DataFrame(data)
    p = tmp_path / "P099_KT88_with_times.csv"
    df.to_csv(p, index=False)
    sig, codes, sid = load_subject(p)
    assert sig.shape[0] == n               # 2 empty rows dropped
    assert not np.isnan(sig).any()
    assert sid == "P099"
