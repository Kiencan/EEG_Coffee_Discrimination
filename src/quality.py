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
    max_dev = float(np.max(np.abs(diffs - median_diff)) / median_diff) if median_diff > 0 else float("inf")
    regular = bool(
        deviating < tol
        and abs(median_fs - fs) < 0.5 * fs
        and max_dev < 2.0
    )
    return {
        "median_fs": median_fs,
        "regular": regular,
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
