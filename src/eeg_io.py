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
