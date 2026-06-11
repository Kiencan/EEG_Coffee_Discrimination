# scripts/02_preprocess_epochs.py
"""Filter, re-reference, drop artifact epochs, and save a clean epoch set."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (DATA_DIR, OUTPUT_DIR, FS, EPOCH_LEN, BANDPASS_LOW,
                        BANDPASS_HIGH, FILTER_ORDER, USE_CAR, FLAT_STD_UV,
                        EPOCH_PTP_PERCENTILE, EXCLUDE_SUBJECTS)
from src.eeg_io import load_subject, extract_epochs
from src.preprocess import bandpass_filter, common_average_reference
from src.quality import epoch_is_artifact


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = sorted(DATA_DIR.glob("P*_KT88_with_times.csv"))

    raw_epochs, raw_labels, raw_subjects = [], [], []
    for f in files:
        sig, codes, sid = load_subject(f)
        if sid in EXCLUDE_SUBJECTS:
            continue
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

    ptp = np.array([np.max(np.ptp(ep, axis=1)) for ep in raw_epochs])
    ptp_thresh = float(np.nanpercentile(ptp, EPOCH_PTP_PERCENTILE))
    keep = np.array([not epoch_is_artifact(ep, ptp_thresh, FLAT_STD_UV)
                     for ep in raw_epochs])

    X = raw_epochs[keep]
    y = raw_labels[keep]
    subjects = raw_subjects[keep]
    out = OUTPUT_DIR / "epochs.npz"
    np.savez_compressed(out, X=X, y=y, subjects=subjects)
    print(f"Kept {keep.sum()}/{len(keep)} epochs after artifact rejection "
          f"(ptp_thresh={ptp_thresh:.1f} uV)")
    print(f"Excluded subjects: {sorted(EXCLUDE_SUBJECTS)}")
    print(f"Saved: {out}  X={X.shape}")
    uniq, cnt = np.unique(y, return_counts=True)
    print("Class balance:", dict(zip(uniq.tolist(), cnt.tolist())))


if __name__ == "__main__":
    main()
