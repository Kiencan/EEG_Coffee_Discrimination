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
    total = np.trapezoid(psd, freqs, axis=-1) + 1e-12            # (n_ch,)
    abs_rows, rel_rows = [], []
    for (lo, hi) in BANDS.values():
        idx = (freqs >= lo) & (freqs < hi)
        bp = np.trapezoid(psd[:, idx], freqs[idx], axis=-1)     # (n_ch,)
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
