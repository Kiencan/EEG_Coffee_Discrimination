"""Linking subjective sensory ratings to EEG trials."""
import numpy as np
import pandas as pd

RATING_COLS = ("valence", "intensity", "favourite")


def load_sensory_long(csv_path):
    """Reshape the wide sensory CSV into long form.

    Wide layout: for person p in 1..20 there are columns Person_p (stimulus
    code) and Valence[.k]/Intensity[.k]/Favourite[.k] with k = p-1 (pandas dedup
    suffix). Returns DataFrame [subject, trial, code, valence, intensity,
    favourite]; subject = 'P0XX' (zero-padded, Person_1 -> 'P001'); rows with
    NaN code are dropped; trial is 1..n in file order.
    """
    df = pd.read_csv(csv_path)
    blocks = []
    for p in range(1, 21):
        suffix = "" if p == 1 else f".{p - 1}"
        cols = {
            f"Person_{p}": "code",
            f"Valence{suffix}": "valence",
            f"Intensity{suffix}": "intensity",
            f"Favourite{suffix}": "favourite",
        }
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        sub = df[list(cols)].rename(columns=cols).copy()
        sub = sub.dropna(subset=["code"]).reset_index(drop=True)
        sub["subject"] = f"P{p:03d}"
        sub["trial"] = np.arange(1, len(sub) + 1)
        sub["code"] = sub["code"].astype(int)
        blocks.append(sub)
    long = pd.concat(blocks, ignore_index=True)
    return long[["subject", "trial", "code", "valence", "intensity",
                 "favourite"]]


def align_ratings(ordered_codes, subject_sensory_df):
    """Align ratings to a sequence of EEG stimulus codes by position.

    ordered_codes : ints in EEG presentation order.
    subject_sensory_df : long rows for ONE subject in trial order.
    Returns dict {'valence','intensity','favourite'} of float arrays aligned to
    ordered_codes. Raises ValueError if the code sequences differ at all.
    """
    ordered_codes = [int(c) for c in ordered_codes]
    sens_codes = subject_sensory_df["code"].astype(int).tolist()
    if ordered_codes != sens_codes:
        raise ValueError(
            f"Code sequence mismatch: {len(ordered_codes)} EEG codes vs "
            f"{len(sens_codes)} sensory codes (or different order)")
    return {col: subject_sensory_df[col].to_numpy(dtype=float)
            for col in RATING_COLS}


def load_trial_order(xlsx_path, n_trials=45):
    """Load the canonical per-subject stimulus order from experiment_sequences.

    Reads sheet "Sheet3": header row at index 7 (Trial, Person_1, ...), data
    from row 8. Returns dict {subject 'P0XX': [code, ...]} with the first
    `n_trials` codes per person (the delivered trials). Persons are zero-padded
    (Person_1 -> 'P001').
    """
    raw = pd.read_excel(xlsx_path, sheet_name="Sheet3", header=None)
    header = [str(x) for x in raw.iloc[7].tolist()]
    order = {}
    for col, name in enumerate(header):
        if not name.startswith("Person_"):
            continue
        num = int(name.split("_")[1])
        codes = pd.to_numeric(raw.iloc[8:, col], errors="coerce").dropna()
        codes = [int(c) for c in codes.tolist()[:n_trials]]
        if codes:
            order[f"P{num:03d}"] = codes
    return order
