# EEG Coffee Arabica vs Robusta — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that checks EEG data quality, extracts clean 3-second coffee-sniff epochs, computes band-power features, and classifies Arabica vs Robusta with subject-independent (LOSO) evaluation.

**Architecture:** Importable modules (`config`, `eeg_io`, `quality`, `preprocess`, `features`, `evaluate`) with thin script entrypoints. Each module is unit-tested with synthetic fixtures (TDD); a final end-to-end script runs the real 18 CSV files. Classification = band power (Welch PSD over delta/theta/alpha/beta/gamma) → StandardScaler → classical classifiers, evaluated with LeaveOneGroupOut across subjects.

**Tech Stack:** Python 3, numpy, scipy, pandas, scikit-learn, matplotlib, pytest.

---

## File Structure

```
machine_learning/
├── src/
│   ├── __init__.py
│   ├── config.py          # codes, bands, fs, thresholds, paths
│   ├── eeg_io.py          # load CSV, run-length, epoch extraction
│   ├── quality.py         # file/channel/epoch quality checks + report
│   ├── preprocess.py      # bandpass filter + common average reference
│   ├── features.py        # Welch band power feature vectors
│   └── evaluate.py        # LOSO cross-validation + metrics
├── scripts/
│   ├── 01_data_quality.py
│   ├── 02_preprocess_epochs.py
│   ├── 03_features.py
│   └── 04_train_loso.py
├── tests/
│   ├── test_eeg_io.py
│   ├── test_quality.py
│   ├── test_preprocess.py
│   ├── test_features.py
│   └── test_evaluate.py
├── outputs/               # generated reports, npz, figures (gitignored)
├── requirements.txt
└── data/                  # existing P0XX_KT88_with_times.csv
```

**Note on git:** the working directory is not yet a git repo. Task 1 initializes it. If you prefer not to use git, skip the `git commit` steps — they are not required for the pipeline to work.

---

## Task 1: Project setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py` (empty)
- Create: `src/config.py`
- Create: `outputs/.gitkeep` (empty)
- Create: `.gitignore`

- [ ] **Step 1: Initialize git and create directories**

```bash
cd "G:/dulieu_23_11_2025/Research/EEG-Coffee/Data/machine_learning"
git init
mkdir -p src scripts tests outputs
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
outputs/
.pytest_cache/
*.npz
```

- [ ] **Step 3: Write `requirements.txt`**

```
numpy
scipy
pandas
scikit-learn
matplotlib
pytest
```

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install successfully.

- [ ] **Step 5: Write `src/__init__.py`**

Empty file.

- [ ] **Step 6: Write `src/config.py`**

```python
"""Central configuration for the EEG coffee classification pipeline."""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# Sampling
FS = 100  # Hz (confirmed: timestamp diff ~0.01s)
EPOCH_LEN = 300  # samples = 3 s sniff window

# Channels (KT88): 16 EEG + 2 ECG dropped
EEG_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
                "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6"]
ECG_CHANNELS = ["ECG1", "ECG2"]

# Stimulus codes (from protocol)
ARABICA_CODES = {981, 633, 902, 598, 733}   # C1
ROBUSTA_CODES = {585, 597, 200, 558, 692}   # C2
CONTROL_CODES = {712, 238, 759, 869, 562}   # C0 (ignored)

# Frequency bands (Hz); high capped < Nyquist (50 Hz)
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Preprocessing
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 45.0
FILTER_ORDER = 4
USE_CAR = True  # common average reference

# Quality thresholds
SATURATION_UV = 204.0      # KT88 appears to clip near +/-204.8 uV
FLAT_STD_UV = 0.5          # channel/epoch with std below this is "flat"
# Epoch artifact threshold calibrated from data; fallback fixed value:
EPOCH_PTP_PERCENTILE = 99  # reject epochs above this ptp percentile
```

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt src/__init__.py src/config.py
git commit -m "chore: project setup and config"
```

---

## Task 2: Code run-length and epoch labeling (`eeg_io` core)

**Files:**
- Create: `src/eeg_io.py`
- Test: `tests/test_eeg_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eeg_io.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eeg_io.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'find_code_runs'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/eeg_io.py
"""Loading EEG CSV files and extracting labeled sniff epochs."""
import numpy as np
import pandas as pd

from src.config import (ARABICA_CODES, ROBUSTA_CODES, EEG_CHANNELS, EPOCH_LEN)


def find_code_runs(codes):
    """Return list of (code, start_index, length) for contiguous equal runs."""
    codes = np.asarray(codes)
    if codes.size == 0:
        return []
    change = np.where(np.diff(codes) != 0)[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [codes.size]))
    return [(int(codes[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]


def label_for_code(code):
    """Map a stimulus code to 'Arabica', 'Robusta', or None (ignored)."""
    if code in ARABICA_CODES:
        return "Arabica"
    if code in ROBUSTA_CODES:
        return "Robusta"
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eeg_io.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/eeg_io.py tests/test_eeg_io.py
git commit -m "feat: code run-length and epoch labeling"
```

---

## Task 3: Epoch extraction and CSV loading (`eeg_io`)

**Files:**
- Modify: `src/eeg_io.py`
- Test: `tests/test_eeg_io.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_eeg_io.py
from src.eeg_io import extract_epochs

def test_extract_epochs_one_per_run():
    # signals: 16 channels, build a session with one Arabica (981) and one
    # Robusta (585) run of EPOCH_LEN samples each, separated by code 0.
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
    assert X.shape == (2, n_ch, EPOCH_LEN)   # (n_epochs, channels, samples)
    assert list(y) == ["Arabica", "Robusta"]
    assert list(run_codes) == [981, 585]
    assert np.allclose(X[0], 1.0) and np.allclose(X[1], 2.0)

def test_extract_epochs_skips_short_and_control():
    from src.config import EPOCH_LEN
    n_ch = 16
    short = np.ones((EPOCH_LEN - 50, n_ch))     # too short Arabica
    ctrl = np.ones((EPOCH_LEN, n_ch))           # control code
    signals = np.vstack([short, ctrl])
    codes = np.concatenate([np.full(EPOCH_LEN - 50, 981),
                            np.full(EPOCH_LEN, 712)])
    X, y, _ = extract_epochs(signals, codes, expected_len=EPOCH_LEN)
    assert X.shape[0] == 0   # short skipped, control ignored
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eeg_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_epochs'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/eeg_io.py

def extract_epochs(signals, codes, expected_len=EPOCH_LEN):
    """Extract coffee-sniff epochs from a session.

    Parameters
    ----------
    signals : ndarray (n_samples, n_channels)
    codes   : ndarray (n_samples,)
    expected_len : int, required epoch length in samples

    Returns
    -------
    X : ndarray (n_epochs, n_channels, expected_len)
    y : ndarray of str labels
    run_codes : ndarray of int stimulus codes
    """
    signals = np.asarray(signals)
    epochs, labels, run_codes = [], [], []
    for code, start, length in find_code_runs(codes):
        label = label_for_code(code)
        if label is None:
            continue
        if length < expected_len:
            continue  # marker truncated; drop
        seg = signals[start:start + expected_len]      # (expected_len, n_ch)
        epochs.append(seg.T)                            # -> (n_ch, expected_len)
        labels.append(label)
        run_codes.append(code)
    if not epochs:
        return (np.empty((0, signals.shape[1], expected_len)),
                np.array([], dtype=object), np.array([], dtype=int))
    return np.stack(epochs), np.array(labels, dtype=object), np.array(run_codes)


def load_subject(csv_path):
    """Load one subject CSV. Returns (signals, codes, subject_id).

    signals : ndarray (n_samples, 16) EEG channels only (ECG dropped)
    codes   : ndarray (n_samples,) int
    subject_id : str e.g. 'P001'
    """
    df = pd.read_csv(csv_path)
    signals = df[EEG_CHANNELS].to_numpy(dtype=float)
    codes = df["code"].to_numpy()
    # codes may contain non-numeric header artifacts; coerce safely
    codes = pd.to_numeric(pd.Series(codes), errors="coerce").fillna(0).astype(int).to_numpy()
    subject_id = str(csv_path).split("/")[-1].split("\\")[-1][:4]
    return signals, codes, subject_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eeg_io.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Smoke-test loader on real data**

Run:
```bash
python -c "from src.eeg_io import load_subject, extract_epochs; from src.config import DATA_DIR; s,c,sid=load_subject(DATA_DIR/'P001_KT88_with_times.csv'); X,y,_=extract_epochs(s,c); print(sid, X.shape, dict(zip(*__import__('numpy').unique(y, return_counts=True))))"
```
Expected: `P001 (30, 16, 300) {'Arabica': 15, 'Robusta': 15}`.

- [ ] **Step 6: Commit**

```bash
git add src/eeg_io.py tests/test_eeg_io.py
git commit -m "feat: epoch extraction and subject CSV loading"
```

---

## Task 4: Data quality checks (`quality`)

**Files:**
- Create: `src/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quality.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.quality'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/quality.py
"""Data quality checks: file, channel, and epoch level."""
import numpy as np


def check_sampling_regularity(timestamps, fs=100, tol=0.2):
    """Check timestamp spacing matches expected sampling period.

    Returns dict with median_fs, regular (bool), and max relative deviation.
    `tol` is the allowed fraction of samples deviating > 50% from median diff.
    """
    ts = np.asarray(timestamps, dtype=float)
    diffs = np.diff(ts)
    median_diff = np.median(diffs)
    median_fs = 1.0 / median_diff if median_diff > 0 else float("inf")
    deviating = np.mean(np.abs(diffs - median_diff) > 0.5 * median_diff)
    return {
        "median_fs": median_fs,
        "regular": bool(deviating < tol and abs(median_fs - fs) < 0.5 * fs),
        "frac_deviating": float(deviating),
    }


def flat_channels(signals, flat_std=0.5):
    """Return indices of channels whose std is below flat_std (dead channels).

    signals : ndarray (n_samples, n_channels)
    """
    stds = np.std(np.asarray(signals, dtype=float), axis=0)
    return [int(i) for i in np.where(stds < flat_std)[0]]


def epoch_is_artifact(epoch, ptp_thresh, flat_std=0.5):
    """True if epoch should be rejected.

    epoch : ndarray (n_channels, n_samples)
    Reject if any channel peak-to-peak exceeds ptp_thresh, or any channel is
    flat (std < flat_std).
    """
    epoch = np.asarray(epoch, dtype=float)
    ptp = np.ptp(epoch, axis=1)
    if np.any(ptp > ptp_thresh):
        return True
    stds = np.std(epoch, axis=1)
    if np.any(stds < flat_std):
        return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_quality.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quality.py tests/test_quality.py
git commit -m "feat: data quality checks"
```

---

## Task 5: Preprocessing — bandpass + CAR (`preprocess`)

**Files:**
- Create: `src/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.preprocess'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/preprocess.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocess.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: bandpass filter and common average reference"
```

---

## Task 6: Band-power features (`features`)

**Files:**
- Create: `src/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import numpy as np
from src.features import band_powers, feature_names
from src.config import BANDS, EEG_CHANNELS

def test_feature_vector_length():
    epoch = np.random.randn(16, 300)
    feats = band_powers(epoch, fs=100)
    # absolute + relative for each (band, channel)
    assert feats.shape == (2 * len(BANDS) * 16,)

def test_feature_names_match_length():
    names = feature_names()
    assert len(names) == 2 * len(BANDS) * len(EEG_CHANNELS)
    assert "abs_alpha_O1" in names and "rel_alpha_O1" in names

def test_alpha_sine_dominates_alpha_band():
    fs = 100
    t = np.arange(0, 3, 1/fs)
    # one channel: pure 10 Hz (alpha) sine, rest noise
    epoch = np.random.randn(16, len(t)) * 0.01
    epoch[0] = np.sin(2*np.pi*10*t)
    feats = band_powers(epoch, fs=fs)
    n = len(BANDS) * 16
    # relative powers are the second half; for channel 0, alpha index:
    band_list = list(BANDS.keys())
    alpha_i = band_list.index("alpha")
    rel = feats[n:].reshape(len(BANDS), 16)
    # alpha relative power for channel 0 should be the max band for that channel
    assert np.argmax(rel[:, 0]) == alpha_i
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.features'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/features.py
"""Band-power feature extraction via Welch PSD."""
import numpy as np
from scipy.signal import welch

from src.config import BANDS, EEG_CHANNELS, FS


def band_powers(epoch, fs=FS):
    """Compute absolute + relative band power per channel.

    epoch : ndarray (n_channels, n_samples)
    Returns 1-D vector: [abs(bands x channels), rel(bands x channels)].
    Layout: for each band (outer), each channel (inner).
    """
    epoch = np.asarray(epoch, dtype=float)
    nperseg = min(256, epoch.shape[-1])
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, axis=-1)  # psd: (n_ch, n_freq)
    total = np.trapz(psd, freqs, axis=-1) + 1e-12               # (n_ch,)
    abs_rows, rel_rows = [], []
    for (lo, hi) in BANDS.values():
        idx = (freqs >= lo) & (freqs < hi)
        bp = np.trapz(psd[:, idx], freqs[idx], axis=-1)         # (n_ch,)
        abs_rows.append(bp)
        rel_rows.append(bp / total)
    abs_bp = np.concatenate(abs_rows)
    rel_bp = np.concatenate(rel_rows)
    return np.concatenate([abs_bp, rel_bp])


def feature_names():
    """Names aligned with band_powers() output order."""
    names = []
    for prefix in ("abs", "rel"):
        for band in BANDS:
            for ch in EEG_CHANNELS:
                names.append(f"{prefix}_{band}_{ch}")
    return names


def build_feature_matrix(X_epochs, fs=FS):
    """Map (n_epochs, n_ch, n_samples) -> (n_epochs, n_features)."""
    return np.vstack([band_powers(ep, fs=fs) for ep in X_epochs])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: Welch band-power features"
```

---

## Task 7: LOSO evaluation (`evaluate`)

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
from src.evaluate import run_loso, make_classifiers

def test_no_subject_leakage_in_folds():
    # 4 subjects, 10 samples each; verify train/test subject sets disjoint
    from sklearn.model_selection import LeaveOneGroupOut
    groups = np.repeat(["A", "B", "C", "D"], 10)
    logo = LeaveOneGroupOut()
    X = np.random.randn(40, 5)
    for tr, te in logo.split(X, np.zeros(40), groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))

def test_run_loso_returns_metrics():
    rng = np.random.RandomState(0)
    # Separable toy data: class depends on feature 0; 4 subjects.
    n_per = 20
    subjects = np.repeat(["S1", "S2", "S3", "S4"], n_per)
    y = np.tile(np.array(["Arabica", "Robusta"] * (n_per // 2)), 4)
    X = rng.randn(len(y), 6)
    X[y == "Arabica", 0] += 3.0   # clear separation
    clf = make_classifiers()["logreg"]
    res = run_loso(X, y, subjects, clf)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["accuracy"] > 0.8                     # separable -> high
    assert set(res["per_subject"].keys()) == {"S1", "S2", "S3", "S4"}
    assert res["confusion_matrix"].shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluate.py
"""Leave-One-Subject-Out cross-validation and metrics."""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)


def make_classifiers():
    """Return dict of named sklearn pipelines (scaler + classifier)."""
    return {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000)),
        "svm_rbf": make_pipeline(StandardScaler(),
                                 SVC(kernel="rbf", probability=True)),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, random_state=0)),
    }


def run_loso(X, y, subjects, clf, positive_label="Arabica"):
    """Leave-One-Subject-Out CV. Returns aggregated metrics.

    X : (n_samples, n_features); y : labels; subjects : group ids.
    clf : an unfitted sklearn estimator/pipeline (cloned per fold).
    """
    from sklearn.base import clone
    X = np.asarray(X)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    logo = LeaveOneGroupOut()

    all_true, all_pred, all_score = [], [], []
    per_subject = {}
    for tr, te in logo.split(X, y, groups=subjects):
        model = clone(clf)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        per_subject[str(subjects[te][0])] = accuracy_score(y[te], pred)
        all_true.extend(y[te])
        all_pred.extend(pred)
        # probability of positive class for AUC, if available
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            pi = classes.index(positive_label)
            all_score.extend(model.predict_proba(X[te])[:, pi])
        else:
            all_score.extend([np.nan] * len(te))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    labels_sorted = sorted(set(y))
    result = {
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro"),
        "confusion_matrix": confusion_matrix(all_true, all_pred,
                                             labels=labels_sorted),
        "labels": labels_sorted,
        "per_subject": per_subject,
    }
    scores = np.array(all_score, dtype=float)
    if not np.any(np.isnan(scores)):
        y_bin = (all_true == positive_label).astype(int)
        result["roc_auc"] = roc_auc_score(y_bin, scores)
    else:
        result["roc_auc"] = float("nan")
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: LOSO cross-validation and metrics"
```

---

## Task 8: Quality report script (`scripts/01_data_quality.py`)

**Files:**
- Create: `scripts/01_data_quality.py`

- [ ] **Step 1: Write the script**

```python
# scripts/01_data_quality.py
"""Generate a data-quality report across all subject CSV files."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (DATA_DIR, OUTPUT_DIR, EEG_CHANNELS, FS, EPOCH_LEN,
                        SATURATION_UV, FLAT_STD_UV, EPOCH_PTP_PERCENTILE)
from src.eeg_io import load_subject, extract_epochs
from src.quality import (check_sampling_regularity, flat_channels,
                         epoch_is_artifact)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = sorted(DATA_DIR.glob("P*_KT88_with_times.csv"))
    rows = []
    # First pass: gather per-epoch ptp to calibrate threshold
    all_ptp = []
    cache = {}
    for f in files:
        sig, codes, sid = load_subject(f)
        X, y, _ = extract_epochs(sig, codes, expected_len=EPOCH_LEN)
        cache[sid] = (sig, codes, X, y, f)
        for ep in X:
            all_ptp.append(np.max(np.ptp(ep, axis=1)))
    ptp_thresh = float(np.percentile(all_ptp, EPOCH_PTP_PERCENTILE)) if all_ptp else 1e9
    print(f"Calibrated epoch ptp threshold ({EPOCH_PTP_PERCENTILE}th pct): {ptp_thresh:.1f} uV")

    for sid, (sig, codes, X, y, f) in cache.items():
        df = pd.read_csv(f, usecols=["timestamp"])
        reg = check_sampling_regularity(df["timestamp"].to_numpy(), fs=FS)
        flats = flat_channels(sig, flat_std=FLAT_STD_UV)
        flat_names = [EEG_CHANNELS[i] for i in flats]
        sat = int(np.sum(np.abs(sig) >= SATURATION_UV))
        n_ara = int(np.sum(y == "Arabica"))
        n_rob = int(np.sum(y == "Robusta"))
        rejected = sum(epoch_is_artifact(ep, ptp_thresh, FLAT_STD_UV) for ep in X)
        rows.append({
            "subject": sid,
            "median_fs": round(reg["median_fs"], 2),
            "sampling_regular": reg["regular"],
            "flat_channels": ",".join(flat_names) if flat_names else "",
            "saturated_samples": sat,
            "n_arabica": n_ara,
            "n_robusta": n_rob,
            "n_epochs_total": len(y),
            "n_epochs_rejected": int(rejected),
            "n_epochs_clean": int(len(y) - rejected),
        })

    report = pd.DataFrame(rows)
    out = OUTPUT_DIR / "quality_report.csv"
    report.to_csv(out, index=False)
    print(report.to_string(index=False))
    print(f"\nSaved: {out}")
    print(f"Total clean epochs: {report['n_epochs_clean'].sum()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script on real data**

Run: `python scripts/01_data_quality.py`
Expected: a printed table with 18 rows (P001..P020 minus P002, P008), each
showing ~15 Arabica / ~15 Robusta, sampling_regular True, and a saved
`outputs/quality_report.csv`. Review flat channels / rejected counts.

- [ ] **Step 3: Commit**

```bash
git add scripts/01_data_quality.py
git commit -m "feat: data quality report script"
```

---

## Task 9: Preprocess + epoch export script (`scripts/02_preprocess_epochs.py`)

**Files:**
- Create: `scripts/02_preprocess_epochs.py`

- [ ] **Step 1: Write the script**

```python
# scripts/02_preprocess_epochs.py
"""Filter, re-reference, drop artifact epochs, and save a clean epoch set."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (DATA_DIR, OUTPUT_DIR, FS, EPOCH_LEN, BANDPASS_LOW,
                        BANDPASS_HIGH, FILTER_ORDER, USE_CAR, FLAT_STD_UV,
                        EPOCH_PTP_PERCENTILE)
from src.eeg_io import load_subject, extract_epochs
from src.preprocess import bandpass_filter, common_average_reference
from src.quality import epoch_is_artifact


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = sorted(DATA_DIR.glob("P*_KT88_with_times.csv"))

    raw_epochs, raw_labels, raw_subjects = [], [], []
    for f in files:
        sig, codes, sid = load_subject(f)
        # filter whole-session signal per channel (axis=0 is samples)
        filt = bandpass_filter(sig.T, fs=FS, low=BANDPASS_LOW,
                               high=BANDPASS_HIGH, order=FILTER_ORDER).T
        X, y, _ = extract_epochs(filt, codes, expected_len=EPOCH_LEN)
        for ep, lab in zip(X, y):
            if USE_CAR:
                ep = common_average_reference(ep)
            raw_epochs.append(ep)
            raw_labels.append(lab)
            raw_subjects.append(sid)

    raw_epochs = np.stack(raw_epochs)
    raw_labels = np.array(raw_labels, dtype=object)
    raw_subjects = np.array(raw_subjects, dtype=object)

    # calibrate ptp threshold on all epochs, then drop artifacts
    ptp = np.array([np.max(np.ptp(ep, axis=1)) for ep in raw_epochs])
    ptp_thresh = float(np.percentile(ptp, EPOCH_PTP_PERCENTILE))
    keep = np.array([not epoch_is_artifact(ep, ptp_thresh, FLAT_STD_UV)
                     for ep in raw_epochs])

    X = raw_epochs[keep]
    y = raw_labels[keep]
    subjects = raw_subjects[keep]
    out = OUTPUT_DIR / "epochs.npz"
    np.savez_compressed(out, X=X, y=y, subjects=subjects)
    print(f"Kept {keep.sum()}/{len(keep)} epochs after artifact rejection "
          f"(ptp_thresh={ptp_thresh:.1f} uV)")
    print(f"Saved: {out}  X={X.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `python scripts/02_preprocess_epochs.py`
Expected: prints kept/total epochs (~near 540 minus a few rejected) and saves
`outputs/epochs.npz` with `X` shape like `(~530, 16, 300)`.

- [ ] **Step 3: Commit**

```bash
git add scripts/02_preprocess_epochs.py
git commit -m "feat: preprocessing and clean epoch export"
```

---

## Task 10: Feature build + LOSO training script (`scripts/03_features.py`, `scripts/04_train_loso.py`)

**Files:**
- Create: `scripts/03_features.py`
- Create: `scripts/04_train_loso.py`

- [ ] **Step 1: Write `scripts/03_features.py`**

```python
# scripts/03_features.py
"""Build the band-power feature matrix from clean epochs."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR, FS
from src.features import build_feature_matrix, feature_names


def main():
    data = np.load(OUTPUT_DIR / "epochs.npz", allow_pickle=True)
    X_ep, y, subjects = data["X"], data["y"], data["subjects"]
    F = build_feature_matrix(X_ep, fs=FS)
    out = OUTPUT_DIR / "features.npz"
    np.savez_compressed(out, F=F, y=y, subjects=subjects,
                        names=np.array(feature_names(), dtype=object))
    print(f"Feature matrix: {F.shape}  ->  {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/03_features.py`
Expected: `Feature matrix: (~530, 160)  ->  outputs/features.npz`.

- [ ] **Step 3: Write `scripts/04_train_loso.py`**

```python
# scripts/04_train_loso.py
"""Run LOSO cross-validation for each classifier and report results."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.evaluate import make_classifiers, run_loso


def main():
    data = np.load(OUTPUT_DIR / "features.npz", allow_pickle=True)
    F, y, subjects = data["F"], data["y"], data["subjects"]
    print(f"Data: {F.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Subjects: {len(np.unique(subjects))}  (chance = 50%)\n")

    summary = []
    best = None
    for name, clf in make_classifiers().items():
        res = run_loso(F, y, subjects, clf)
        summary.append({
            "model": name,
            "accuracy": round(res["accuracy"], 3),
            "macro_f1": round(res["macro_f1"], 3),
            "roc_auc": round(res["roc_auc"], 3),
        })
        print(f"{name:14s} acc={res['accuracy']:.3f} f1={res['macro_f1']:.3f} "
              f"auc={res['roc_auc']:.3f}")
        if best is None or res["accuracy"] > best[1]["accuracy"]:
            best = (name, res)

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "loso_results.csv", index=False)

    # Confusion matrix + per-subject accuracy for best model
    name, res = best
    cm = res["confusion_matrix"]
    labels = res["labels"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    im = ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels)
    ax[0].set_yticks(range(len(labels))); ax[0].set_yticklabels(labels)
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
    ax[0].set_title(f"Confusion ({name})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax[0].text(j, i, cm[i, j], ha="center", va="center")
    subs = list(res["per_subject"].keys())
    accs = [res["per_subject"][s] for s in subs]
    ax[1].bar(range(len(subs)), accs)
    ax[1].axhline(0.5, color="r", ls="--", label="chance")
    ax[1].set_xticks(range(len(subs))); ax[1].set_xticklabels(subs, rotation=90)
    ax[1].set_ylabel("Accuracy"); ax[1].set_title("Per-subject accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loso_figures.png", dpi=120)
    print(f"\nBest: {name}. Saved loso_results.csv and loso_figures.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it**

Run: `python scripts/04_train_loso.py`
Expected: prints accuracy/F1/AUC for logreg, svm_rbf, random_forest; saves
`outputs/loso_results.csv` and `outputs/loso_figures.png`. Accuracy near 50%
means the brain signal carries little Arabica/Robusta information under LOSO;
notably above 50% indicates a learnable subject-independent effect.

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests across the 5 test files PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/03_features.py scripts/04_train_loso.py
git commit -m "feat: feature build and LOSO training scripts"
```

---

## Self-Review Notes

- **Spec coverage:** quality checks (Task 4, 8), preprocessing 1–45 Hz + CAR
  (Task 5, 9), band power abs+rel features (Task 6, 10), LOSO with scaler-in-fold
  and full metrics (Task 7, 10), modular structure + scripts (all). Control codes
  ignored via `label_for_code` returning None. ECG dropped in `load_subject`.
- **Calibrated artifact threshold:** implemented as a data-driven percentile
  (Task 8 prints it; Task 9 applies it) per spec's "calibrate from data".
- **Saturation note:** `SATURATION_UV=204` flags KT88 clipping in the report.
- **Determinism:** RandomForest and toy tests use fixed seeds.
```