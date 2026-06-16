"""Loading EEG CSV files and extracting labeled sniff epochs."""
import numpy as np
import pandas as pd

from src.config import (ARABICA_CODES, ROBUSTA_CODES, CONTROL_CODES,
                        EEG_CHANNELS, EPOCH_LEN)


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


def label_for_task(code, task="coffee"):
    """Map a stimulus code to a label for a given classification task.

    task="coffee"            -> Arabica / Robusta / None (original behavior).
    task="control_vs_coffee" -> Control (C0) / Coffee (C1 or C2) / None.
    """
    if task == "coffee":
        return label_for_code(code)
    if task == "control_vs_coffee":
        if code in CONTROL_CODES:
            return "Control"
        if code in ARABICA_CODES or code in ROBUSTA_CODES:
            return "Coffee"
        return None
    raise ValueError(f"Unknown task: {task}")


def extract_epochs_task(signals, codes, task="coffee", expected_len=EPOCH_LEN):
    """Extract labeled epochs for a given task (see label_for_task)."""
    signals = np.asarray(signals)
    epochs, labels, run_codes = [], [], []
    for code, start, length in find_code_runs(codes):
        label = label_for_task(code, task=task)
        if label is None:
            continue
        if length < expected_len:
            continue
        seg = signals[start:start + expected_len]
        epochs.append(seg.T)
        labels.append(label)
        run_codes.append(code)
    if not epochs:
        return (np.empty((0, signals.shape[1], expected_len)),
                np.array([], dtype=object), np.array([], dtype=int))
    return np.stack(epochs), np.array(labels, dtype=object), np.array(run_codes)


def extract_epochs(signals, codes, expected_len=EPOCH_LEN):
    """Extract coffee-sniff epochs (Arabica/Robusta). Thin wrapper kept for
    backward compatibility; delegates to extract_epochs_task(task="coffee")."""
    return extract_epochs_task(signals, codes, task="coffee",
                               expected_len=expected_len)


def load_subject(csv_path):
    """Load one subject CSV. Returns (signals, codes, subject_id).

    signals : ndarray (n_samples, 16) EEG channels only (ECG dropped)
    codes   : ndarray (n_samples,) int
    subject_id : str e.g. 'P001'

    Rows where all EEG channels are NaN (e.g. trailing empty rows) are dropped
    so they cannot propagate NaN through filtering.
    """
    df = pd.read_csv(csv_path)
    eeg_nan_all = df[EEG_CHANNELS].isna().all(axis=1)
    if eeg_nan_all.any():
        df = df[~eeg_nan_all].reset_index(drop=True)
    signals = df[EEG_CHANNELS].to_numpy(dtype=float)
    codes = df["code"].to_numpy()
    # codes may contain non-numeric header artifacts; coerce safely
    codes = pd.to_numeric(pd.Series(codes), errors="coerce").fillna(0).astype(int).to_numpy()
    subject_id = str(csv_path).split("/")[-1].split("\\")[-1][:4]
    return signals, codes, subject_id
