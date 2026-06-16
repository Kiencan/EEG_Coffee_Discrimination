# scripts/11_build_c0_vs_coffee.py
"""Build Control (C0) vs Coffee (C1 union C2) epoch set from clean subjects."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (DATA_DIR, OUTPUT_DIR, FS, EPOCH_LEN, BANDPASS_LOW,
                        BANDPASS_HIGH, FILTER_ORDER, USE_CAR, FLAT_STD_UV,
                        EPOCH_PTP_PERCENTILE, EXCLUDE_SUBJECTS)
from src.eeg_io import load_subject, extract_epochs_task
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
        X, y, _ = extract_epochs_task(filt, codes, task="control_vs_coffee",
                                      expected_len=EPOCH_LEN)
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
    out = OUTPUT_DIR / "epochs_c0_vs_coffee.npz"
    np.savez_compressed(out, X=X, y=y, subjects=subjects)
    print(f"Kept {keep.sum()}/{len(keep)} epochs (ptp_thresh={ptp_thresh:.1f} uV)")
    print(f"Subjects ({len(set(subjects.tolist()))}): {sorted(set(subjects.tolist()))}")
    uniq, cnt = np.unique(y, return_counts=True)
    print("Class balance:", dict(zip(uniq.tolist(), cnt.tolist())))
    print(f"Saved: {out}  X={X.shape}")


if __name__ == "__main__":
    main()
