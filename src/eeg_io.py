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
