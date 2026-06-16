# scripts/16_build_sensory_dataset.py
"""Link sensory ratings to clean coffee EEG epochs (per-trial).

Subjects whose EEG stimulus order does not match the sensory sheet (e.g. P019)
are skipped with a warning, since their per-trial ratings cannot be aligned.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (ROOT, DATA_DIR, OUTPUT_DIR, FS, EPOCH_LEN, BANDPASS_LOW,
                        BANDPASS_HIGH, FILTER_ORDER, USE_CAR, FLAT_STD_UV,
                        EPOCH_PTP_PERCENTILE, EXCLUDE_SUBJECTS,
                        ARABICA_CODES, ROBUSTA_CODES, CONTROL_CODES)
from src.eeg_io import load_subject, find_code_runs, label_for_code
from src.preprocess import bandpass_filter, common_average_reference
from src.quality import epoch_is_artifact
from src.sensory import load_sensory_long, align_ratings

STIM = ARABICA_CODES | ROBUSTA_CODES | CONTROL_CODES


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    sens = load_sensory_long(ROOT / "protocol" / "sensory_data.csv")
    files = sorted(DATA_DIR.glob("P*_KT88_with_times.csv"))
    Xs, subs, codes_out, conds, val, inten, fav = [], [], [], [], [], [], []
    skipped = []
    for f in files:
        sig, codes, sid = load_subject(f)
        if sid in EXCLUDE_SUBJECTS:
            continue
        filt = bandpass_filter(sig.T, fs=FS, low=BANDPASS_LOW,
                               high=BANDPASS_HIGH, order=FILTER_ORDER).T
        runs = [(c, s, l) for c, s, l in find_code_runs(codes)
                if c in STIM and l >= EPOCH_LEN]
        ordered_codes = [c for c, s, l in runs]
        subject_sens = sens[sens["subject"] == sid].sort_values("trial")
        try:
            ratings = align_ratings(ordered_codes, subject_sens)
        except ValueError as exc:
            skipped.append(sid)
            print(f"WARNING: skipping {sid} -- {exc}")
            continue
        for (c, start, length), v, it, fv in zip(
                runs, ratings["valence"], ratings["intensity"],
                ratings["favourite"]):
            cond = label_for_code(c)         # Arabica/Robusta/None(control)
            if cond is None:
                continue                     # coffee only
            ep = filt[start:start + EPOCH_LEN].T
            if USE_CAR:
                ep = common_average_reference(ep)
            Xs.append(ep); subs.append(sid); codes_out.append(c)
            conds.append(cond); val.append(v); inten.append(it); fav.append(fv)

    X = np.stack(Xs)
    subs = np.array(subs, dtype=object)
    ptp = np.array([np.max(np.ptp(ep, axis=1)) for ep in X])
    ptp_thresh = float(np.nanpercentile(ptp, EPOCH_PTP_PERCENTILE))
    keep = np.array([not epoch_is_artifact(ep, ptp_thresh, FLAT_STD_UV)
                     for ep in X])
    out = OUTPUT_DIR / "epochs_sensory.npz"
    np.savez_compressed(
        out, X=X[keep], subjects=subs[keep],
        codes=np.array(codes_out)[keep],
        condition=np.array(conds, dtype=object)[keep],
        valence=np.array(val)[keep], intensity=np.array(inten)[keep],
        favourite=np.array(fav)[keep])
    print(f"Skipped (misaligned): {skipped}")
    print(f"Kept {keep.sum()}/{len(keep)} coffee epochs "
          f"(ptp_thresh={ptp_thresh:.1f} uV)")
    print(f"Subjects: {sorted(set(subs[keep].tolist()))}")
    print(f"Saved: {out}  X={X[keep].shape}")
    for name, arr in [("valence", val), ("intensity", inten),
                      ("favourite", fav)]:
        a = np.array(arr)[keep]
        print(f"{name}: mean={a.mean():.2f} range=[{a.min():.0f},{a.max():.0f}]")


if __name__ == "__main__":
    main()
