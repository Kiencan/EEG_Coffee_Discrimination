"""Hand-engineered EEG feature families for feature-search experiments."""
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import BANDS, EEG_CHANNELS, FS


NARROW_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "low_alpha": (8, 10),
    "high_alpha": (10, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
    "low_gamma": (30, 40),
    "high_gamma": (40, 45),
}

TIME_FEATURES = (
    "mean",
    "std",
    "rms",
    "ptp",
    "skew",
    "kurtosis",
    "line_length",
    "zero_crossings",
)

HJORTH_FEATURES = ("hjorth_activity", "hjorth_mobility", "hjorth_complexity")


def _channel_names(n_channels):
    if n_channels == len(EEG_CHANNELS):
        return EEG_CHANNELS
    return [f"ch{i:02d}" for i in range(n_channels)]


def _safe_var(x, axis=-1):
    return np.var(np.asarray(x, dtype=float), axis=axis)


def _safe_skew(x):
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1) + 1e-12
    return np.mean(centered ** 3, axis=-1) / (std ** 3)


def _safe_kurtosis(x):
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1) + 1e-12
    return np.mean(centered ** 4, axis=-1) / (std ** 4) - 3.0


def time_domain_features(epoch):
    """Return per-channel time-domain summaries for one epoch."""
    epoch = np.asarray(epoch, dtype=float)
    if epoch.ndim != 2:
        raise ValueError("epoch must have shape (n_channels, n_samples)")
    signs = np.signbit(epoch)
    rows = [
        epoch.mean(axis=-1),
        epoch.std(axis=-1),
        np.sqrt(np.mean(epoch ** 2, axis=-1)),
        np.ptp(epoch, axis=-1),
        _safe_skew(epoch),
        _safe_kurtosis(epoch),
        np.sum(np.abs(np.diff(epoch, axis=-1)), axis=-1),
        np.sum(signs[:, 1:] != signs[:, :-1], axis=-1),
    ]
    return np.concatenate(rows)


def hjorth_features(epoch):
    """Return Hjorth activity, mobility, and complexity per channel."""
    epoch = np.asarray(epoch, dtype=float)
    if epoch.ndim != 2:
        raise ValueError("epoch must have shape (n_channels, n_samples)")
    dx = np.diff(epoch, axis=-1)
    ddx = np.diff(dx, axis=-1)
    activity = _safe_var(epoch)
    mobility = np.sqrt(_safe_var(dx) / (activity + 1e-12))
    mobility_dx = np.sqrt(_safe_var(ddx) / (_safe_var(dx) + 1e-12))
    complexity = mobility_dx / (mobility + 1e-12)
    rows = [activity, mobility, complexity]
    return np.nan_to_num(np.concatenate(rows), nan=0.0, posinf=0.0, neginf=0.0)


def band_power_features(epoch, bands=None, fs=FS, n_windows=1):
    """Return absolute and relative Welch band powers over temporal windows."""
    epoch = np.asarray(epoch, dtype=float)
    if epoch.ndim != 2:
        raise ValueError("epoch must have shape (n_channels, n_samples)")
    bands = BANDS if bands is None else bands
    rows = []
    for window in np.array_split(epoch, n_windows, axis=-1):
        nperseg = min(256, window.shape[-1])
        freqs, psd = welch(window, fs=fs, nperseg=nperseg, axis=-1)
        total = np.trapezoid(psd, freqs, axis=-1) + 1e-12
        abs_rows, rel_rows = [], []
        for lo, hi in bands.values():
            idx = (freqs >= lo) & (freqs < hi)
            bp = np.trapezoid(psd[:, idx], freqs[idx], axis=-1)
            abs_rows.append(bp)
            rel_rows.append(bp / total)
        rows.extend(abs_rows)
        rows.extend(rel_rows)
    return np.concatenate(rows)


def feature_family_names(family, n_channels=len(EEG_CHANNELS)):
    """Return feature names aligned with build_feature_family_matrix()."""
    ch_names = _channel_names(n_channels)
    if family == "time":
        return [f"{feat}_{ch}" for feat in TIME_FEATURES for ch in ch_names]
    if family == "hjorth":
        return [f"{feat}_{ch}" for feat in HJORTH_FEATURES for ch in ch_names]
    if family == "bandpower":
        return _band_power_names(BANDS, ch_names, n_windows=1)
    if family == "temporal_bandpower":
        return _band_power_names(BANDS, ch_names, n_windows=3)
    if family == "narrow_bandpower":
        return _band_power_names(NARROW_BANDS, ch_names, n_windows=1)
    if family == "engineered":
        names = []
        for fam in ("time", "hjorth", "temporal_bandpower", "narrow_bandpower"):
            names.extend(feature_family_names(fam, n_channels=n_channels))
        return names
    raise ValueError(f"Unknown feature family: {family}")


def _band_power_names(bands, ch_names, n_windows):
    names = []
    for wi in range(n_windows):
        prefix = f"w{wi + 1}_" if n_windows > 1 else ""
        for kind in ("abs", "rel"):
            for band in bands:
                for ch in ch_names:
                    names.append(f"{prefix}{kind}_{band}_{ch}")
    return names


def _feature_vector(epoch, family):
    if family == "time":
        return time_domain_features(epoch)
    if family == "hjorth":
        return hjorth_features(epoch)
    if family == "bandpower":
        return band_power_features(epoch, bands=BANDS, n_windows=1)
    if family == "temporal_bandpower":
        return band_power_features(epoch, bands=BANDS, n_windows=3)
    if family == "narrow_bandpower":
        return band_power_features(epoch, bands=NARROW_BANDS, n_windows=1)
    if family == "engineered":
        return np.concatenate([
            time_domain_features(epoch),
            hjorth_features(epoch),
            band_power_features(epoch, bands=BANDS, n_windows=3),
            band_power_features(epoch, bands=NARROW_BANDS, n_windows=1),
        ])
    raise ValueError(f"Unknown feature family: {family}")


def build_feature_family_matrix(X_epochs, family="engineered"):
    """Build a feature matrix and aligned names for a named feature family."""
    X_epochs = np.asarray(X_epochs, dtype=float)
    if X_epochs.ndim != 3:
        raise ValueError("X_epochs must have shape (n_epochs, n_channels, n_samples)")
    F = np.vstack([_feature_vector(epoch, family) for epoch in X_epochs])
    names = feature_family_names(family, n_channels=X_epochs.shape[1])
    if F.shape[1] != len(names):
        raise RuntimeError("Feature values and names are misaligned")
    return F, names


def make_selected_feature_classifiers(k=80):
    """Return sklearn pipelines with univariate selection inside each fold."""
    return {
        "selected_logreg": make_pipeline(
            StandardScaler(),
            SelectKBest(score_func=f_classif, k=k),
            LogisticRegression(max_iter=2000, random_state=0),
        ),
        "selected_linear_svm": make_pipeline(
            StandardScaler(),
            SelectKBest(score_func=f_classif, k=k),
            LinearSVC(max_iter=5000, random_state=0),
        ),
        "selected_random_forest": make_pipeline(
            StandardScaler(),
            SelectKBest(score_func=f_classif, k=k),
            RandomForestClassifier(n_estimators=300, random_state=0),
        ),
    }
