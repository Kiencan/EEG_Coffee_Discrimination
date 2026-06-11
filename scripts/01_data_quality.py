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
