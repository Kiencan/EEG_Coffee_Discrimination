# scripts/16_build_sensory_dataset.py
"""Link sensory ratings to clean coffee EEG epochs (per-trial), labeling each
time-ordered epoch by the CANONICAL trial order in experiment_sequences.xlsx.

For 17 subjects the canonical order equals the EEG `code` markers. For P019 the
EEG markers are scrambled, so labeling by the canonical order (which matches its
sensory ratings) is required to align epochs with ratings. This trusts the
experiment_sequences.xlsx plan as authoritative: the k-th time-ordered epoch is
the k-th planned stimulus.
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
from src.sensory import load_sensory_long, load_trial_order

STIM = ARABICA_CODES | ROBUSTA_CODES | CONTROL_CODES


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    sens = load_sensory_long(ROOT / "protocol" / "sensory_data.csv")
    order = load_trial_order(ROOT / "protocol" / "experiment_sequences.xlsx")
    files = sorted(DATA_DIR.glob("P*_KT88_with_times.csv"))
    Xs, subs, codes_out, conds, val, inten, fav = [], [], [], [], [], [], []
    skipped = []
    for f in files:
        sig, codes, sid = load_subject(f)
        if sid in EXCLUDE_SUBJECTS:
            continue
        if sid not in order:
            skipped.append((sid, "no canonical order"))
            continue
        canon = order[sid]
        subject_sens = sens[sens["subject"] == sid].sort_values("trial")
        filt = bandpass_filter(sig.T, fs=FS, low=BANDPASS_LOW,
                               high=BANDPASS_HIGH, order=FILTER_ORDER).T
        runs = [(c, s, l) for c, s, l in find_code_runs(codes)
                if c in STIM and l >= EPOCH_LEN]
        # sanity: time-ordered epoch count must equal canonical & sensory length
        if not (len(runs) == len(canon) == len(subject_sens)):
            skipped.append((sid, f"length mismatch runs={len(runs)} "
                                 f"canon={len(canon)} sens={len(subject_sens)}"))
            continue
        # sensory ratings are in canonical order (verified) -> index by position
        v_arr = subject_sens["valence"].to_numpy(dtype=float)
        i_arr = subject_sens["intensity"].to_numpy(dtype=float)
        f_arr = subject_sens["favourite"].to_numpy(dtype=float)
        for k, (mc, start, length) in enumerate(runs):
            code = canon[k]                  # canonical label overrides marker
            cond = label_for_code(code)      # Arabica/Robusta/None(control)
            if cond is None:
                continue                     # coffee only
            ep = filt[start:start + EPOCH_LEN].T
            if USE_CAR:
                ep = common_average_reference(ep)
            Xs.append(ep); subs.append(sid); codes_out.append(code)
            conds.append(cond)
            val.append(v_arr[k]); inten.append(i_arr[k]); fav.append(f_arr[k])

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
    print(f"Skipped: {skipped}")
    print(f"Kept {keep.sum()}/{len(keep)} coffee epochs "
          f"(ptp_thresh={ptp_thresh:.1f} uV)")
    print(f"Subjects ({len(set(subs[keep].tolist()))}): "
          f"{sorted(set(subs[keep].tolist()))}")
    print(f"Saved: {out}  X={X[keep].shape}")
    for name, arr in [("valence", val), ("intensity", inten),
                      ("favourite", fav)]:
        a = np.array(arr)[keep]
        print(f"{name}: mean={a.mean():.2f} range=[{a.min():.0f},{a.max():.0f}]")


if __name__ == "__main__":
    main()
