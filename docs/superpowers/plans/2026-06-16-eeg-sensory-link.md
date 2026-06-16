# EEG ↔ Sensory-rating Link Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Link per-trial subjective sensory ratings (Valence/Intensity/Favourite) to clean coffee EEG epochs and test EEG↔rating association via correlation, regression, and 3-level classification, both within-subject and cross-subject.

**Architecture:** A tested `src/sensory.py` reshapes the wide sensory CSV and aligns ratings to EEG stimulus order by position. A build script produces `outputs/epochs_sensory.npz` (coffee epochs + 3 ratings each). Three analysis scripts (correlation, regression, classification) consume it and write CSV summaries. EEG features = band power (reuse `src/features.py`).

**Tech Stack:** Python 3.14, NumPy 2.x, scipy, scikit-learn, pandas, matplotlib, pytest.

---

## File Structure

```
src/sensory.py                          # load_sensory_long, align_ratings (tested)
tests/test_sensory.py                   # unit tests
scripts/16_build_sensory_dataset.py     # -> outputs/epochs_sensory.npz
scripts/17_sensory_correlation.py       # within-subject Spearman -> CSV
scripts/18_sensory_regression.py        # RidgeCV within+cross -> CSV
scripts/19_sensory_classification.py    # 3-level tertile within+cross -> CSV
```

Work on `master`. Reuse existing modules: `src/eeg_io.py` (load_subject,
find_code_runs, label_for_code), `src/preprocess.py`, `src/quality.py`,
`src/features.py`, `src/config.py` (has ROOT, EXCLUDE_SUBJECTS, code sets).

---

## Task 1: sensory module (load + align)

**Files:**
- Create: `src/sensory.py`
- Test: `tests/test_sensory.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_sensory.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.sensory import load_sensory_long, align_ratings
from src.config import ROOT


def test_load_sensory_long_real_file():
    long = load_sensory_long(ROOT / "protocol" / "sensory_data.csv")
    for col in ["subject", "trial", "code", "valence", "intensity", "favourite"]:
        assert col in long.columns
    assert long["subject"].nunique() == 20
    assert (long.groupby("subject").size() == 45).all()
    assert long["valence"].between(1, 7).all()
    assert long["subject"].iloc[0] == "P001"


def test_align_ratings_match():
    dfp = pd.DataFrame({
        "subject": ["P001"] * 3, "trial": [1, 2, 3],
        "code": [981, 585, 712], "valence": [5, 6, 4],
        "intensity": [6, 5, 1], "favourite": [5, 6, 4]})
    out = align_ratings([981, 585, 712], dfp)
    assert list(out["valence"]) == [5, 6, 4]
    assert list(out["intensity"]) == [6, 5, 1]


def test_align_ratings_mismatch_raises():
    dfp = pd.DataFrame({
        "subject": ["P001"] * 2, "trial": [1, 2], "code": [981, 585],
        "valence": [5, 6], "intensity": [6, 5], "favourite": [5, 6]})
    with pytest.raises(ValueError):
        align_ratings([981, 712], dfp)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sensory.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'src.sensory').

- [ ] **Step 3: Implement** — `src/sensory.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_sensory.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full suite**

Run: `python -m pytest -q`
Expected: all pass (was 37, now 40).

- [ ] **Step 6: Commit**

```bash
git add src/sensory.py tests/test_sensory.py
git commit -m "feat: sensory CSV loader and per-trial rating alignment"
```

---

## Task 2: build sensory-linked dataset

**Files:**
- Create: `scripts/16_build_sensory_dataset.py`

- [ ] **Step 1: Write the script**

```python
# scripts/16_build_sensory_dataset.py
"""Link sensory ratings to clean coffee EEG epochs (per-trial)."""
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
        ratings = align_ratings(ordered_codes, subject_sens)  # raises if mismatch
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
```

- [ ] **Step 2: Run it**

Run: `python scripts/16_build_sensory_dataset.py`
Expected: 8 clean subjects, ~235 coffee epochs kept, prints mean ratings, saves
`outputs/epochs_sensory.npz`. If `align_ratings` raises a mismatch, STOP and
report BLOCKED (it means EEG/sensory order disagree for some subject).

- [ ] **Step 3: Sanity-check the saved file**

Run:
```bash
python -c "import numpy as np; d=np.load('outputs/epochs_sensory.npz', allow_pickle=True); print(list(d.keys())); print('X', d['X'].shape, 'subjects', len(np.unique(d['subjects']))); print('any nan X', np.isnan(d['X'].astype(float)).any()); print('intensity mean', d['intensity'].mean())"
```
Expected: keys include X, subjects, codes, condition, valence, intensity,
favourite; ~235 epochs, 8 subjects, no NaN.

- [ ] **Step 4: Commit**

```bash
git add scripts/16_build_sensory_dataset.py
git commit -m "feat: per-trial sensory-linked coffee epoch dataset"
```

---

## Task 3: within-subject correlation

**Files:**
- Create: `scripts/17_sensory_correlation.py`

- [ ] **Step 1: Write the script**

```python
# scripts/17_sensory_correlation.py
"""Within-subject Spearman correlation: band-power features vs ratings."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.features import build_feature_matrix, feature_names

RATINGS = ["valence", "intensity", "favourite"]


def main():
    data = np.load(OUTPUT_DIR / "epochs_sensory.npz", allow_pickle=True)
    X = data["X"]
    subjects = data["subjects"]
    F = build_feature_matrix(X)
    names = feature_names()
    rows = []
    for rating in RATINGS:
        y = data[rating].astype(float)
        acc = np.zeros(F.shape[1])
        counts = np.zeros(F.shape[1])
        for sid in np.unique(subjects):
            m = subjects == sid
            ys = y[m]
            if m.sum() < 5 or np.std(ys) == 0:
                continue
            for j in range(F.shape[1]):
                xj = F[m, j]
                if np.std(xj) == 0:
                    continue
                r, _ = spearmanr(xj, ys)
                if np.isfinite(r):
                    acc[j] += r
                    counts[j] += 1
        mean_corr = np.divide(acc, counts, out=np.zeros_like(acc),
                              where=counts > 0)
        for j in range(F.shape[1]):
            rows.append({"rating": rating, "feature": names[j],
                         "mean_within_corr": round(float(mean_corr[j]), 4),
                         "n_subjects": int(counts[j])})

    res = pd.DataFrame(rows)
    out = OUTPUT_DIR / "sensory_correlations.csv"
    res.to_csv(out, index=False)
    for rating in RATINGS:
        sub = res[res.rating == rating].copy()
        sub["abs"] = sub["mean_within_corr"].abs()
        top = sub.sort_values("abs", ascending=False).head(8)
        print(f"\nTop 8 features vs {rating} (within-subject mean Spearman):")
        print(top[["feature", "mean_within_corr", "n_subjects"]]
              .to_string(index=False))
    print("\nSummary (max |mean within-subject corr| per rating):")
    for rating in RATINGS:
        sub = res[res.rating == rating]
        print(f"  {rating}: {sub['mean_within_corr'].abs().max():.3f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/17_sensory_correlation.py`
Expected: prints top-8 features per rating + a per-rating max |corr| summary,
saves `outputs/sensory_correlations.csv` (480 rows = 160 features × 3 ratings).
Report the max |corr| honestly (within-subject correlations near 0 mean no
linear association; |corr| > ~0.2 averaged across subjects is notable).

- [ ] **Step 3: Commit**

```bash
git add scripts/17_sensory_correlation.py
git commit -m "feat: within-subject EEG-rating correlation analysis"
```

---

## Task 4: regression (within + cross subject)

**Files:**
- Create: `scripts/18_sensory_regression.py`

- [ ] **Step 1: Write the script**

```python
# scripts/18_sensory_regression.py
"""Predict continuous sensory ratings from band-power EEG (within + cross)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.features import build_feature_matrix

RATINGS = ["valence", "intensity", "favourite"]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def _pipe():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def _within_subject(F, y, subjects):
    preds = np.full(len(y), np.nan)
    for sid in np.unique(subjects):
        m = np.where(subjects == sid)[0]
        if len(m) < 10:
            continue
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in kf.split(m):
            model = _pipe()
            model.fit(F[m[tr]], y[m[tr]])
            preds[m[te]] = model.predict(F[m[te]])
    ok = np.isfinite(preds)
    return (r2_score(y[ok], preds[ok]),
            pearsonr(y[ok], preds[ok])[0], int(ok.sum()))


def _cross_subject(F, y, subjects):
    logo = LeaveOneGroupOut()
    preds = np.full(len(y), np.nan)
    for tr, te in logo.split(F, y, groups=subjects):
        model = _pipe()
        model.fit(F[tr], y[tr])
        preds[te] = model.predict(F[te])
    return r2_score(y, preds), pearsonr(y, preds)[0], len(y)


def main():
    data = np.load(OUTPUT_DIR / "epochs_sensory.npz", allow_pickle=True)
    X = data["X"]
    subjects = data["subjects"]
    F = build_feature_matrix(X)
    rows = []
    for rating in RATINGS:
        y = data[rating].astype(float)
        for level, fn in [("within_subject", _within_subject),
                          ("cross_subject", _cross_subject)]:
            r2, corr, n = fn(F, y, subjects)
            rows.append({"rating": rating, "level": level,
                         "r2": round(float(r2), 3),
                         "pearson_r": round(float(corr), 3), "n": int(n)})
            print(f"{rating:10s} {level:14s} R2={r2:+.3f} r={corr:+.3f} n={n}")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sensory_regression.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'sensory_regression.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/18_sensory_regression.py`
Expected: 6 lines (3 ratings × within/cross) with R² and Pearson r, saves
`outputs/sensory_regression.csv`. Report numbers honestly: R² ≤ 0 means the
model does not predict the rating better than the mean (expected for small
within-subject n and cross-subject baseline differences).

- [ ] **Step 3: Commit**

```bash
git add scripts/18_sensory_regression.py
git commit -m "feat: RidgeCV regression of sensory ratings from EEG"
```

---

## Task 5: 3-level classification (within + cross subject)

**Files:**
- Create: `scripts/19_sensory_classification.py`

- [ ] **Step 1: Write the script**

```python
# scripts/19_sensory_classification.py
"""3-level (low/med/high) sensory-rating classification from band-power EEG."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.features import build_feature_matrix

RATINGS = ["valence", "intensity", "favourite"]


def _tertile_labels(values):
    """Map continuous values to 0/1/2 by tertiles (low/med/high)."""
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    lab = np.zeros(len(values), dtype=int)
    lab[values > q1] = 1
    lab[values > q2] = 2
    return lab


def _rf():
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(n_estimators=300, random_state=0,
                                                class_weight="balanced"))


def _logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000,
                                            class_weight="balanced"))


FACTORIES = {"rf": _rf, "logreg": _logreg}


def _within(F, values, subjects, factory):
    true_all, pred_all = [], []
    for sid in np.unique(subjects):
        m = np.where(subjects == sid)[0]
        if len(m) < 12:
            continue
        lab = _tertile_labels(values[m])         # per-subject tertiles
        if len(np.unique(lab)) < 2:
            continue
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        for tr, te in skf.split(m, lab):
            model = factory()
            model.fit(F[m[tr]], lab[tr])
            true_all.extend(lab[te])
            pred_all.extend(model.predict(F[m[te]]))
    return balanced_accuracy_score(true_all, pred_all), len(true_all)


def _cross(F, values, subjects, factory):
    lab = _tertile_labels(values)                # global tertiles
    logo = LeaveOneGroupOut()
    true_all, pred_all = [], []
    for tr, te in logo.split(F, lab, groups=subjects):
        model = factory()
        model.fit(F[tr], lab[tr])
        true_all.extend(lab[te])
        pred_all.extend(model.predict(F[te]))
    return balanced_accuracy_score(true_all, pred_all), len(true_all)


def main():
    data = np.load(OUTPUT_DIR / "epochs_sensory.npz", allow_pickle=True)
    X = data["X"]
    subjects = data["subjects"]
    F = build_feature_matrix(X)
    rows = []
    for rating in RATINGS:
        values = data[rating].astype(float)
        for model_name, factory in FACTORIES.items():
            for level, fn in [("within_subject", _within),
                              ("cross_subject", _cross)]:
                bal, n = fn(F, values, subjects, factory)
                rows.append({"rating": rating, "model": model_name,
                             "level": level, "balanced_acc": round(bal, 3),
                             "n": n})
                print(f"{rating:10s} {model_name:7s} {level:14s} "
                      f"bal_acc={bal:.3f} (chance=0.333) n={n}")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sensory_classification.csv",
                              index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'sensory_classification.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/19_sensory_classification.py`
Expected: 12 lines (3 ratings × 2 models × within/cross) with balanced accuracy,
saves `outputs/sensory_classification.csv`. Chance = 0.333. Report honestly.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest -q`
Expected: all 40 tests pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/19_sensory_classification.py
git commit -m "feat: 3-level sensory-rating classification from EEG"
```

---

## Self-Review Notes

- **Spec coverage:** sensory module + alignment (Task 1), per-trial dataset
  build coffee-only on clean 8 (Task 2), within-subject correlation all 3
  ratings (Task 3), regression within+cross all 3 (Task 4), 3-level tertile
  classification within+cross all 3 (Task 5). Heatmap dropped in favor of CSV +
  console top-features (lighter; documented deviation).
- **Type consistency:** `align_ratings` returns dict with keys valence/intensity/
  favourite (RATING_COLS); npz keys X/subjects/codes/condition/valence/intensity/
  favourite used identically in Tasks 2–5.
- **No leakage:** scaler/RidgeCV/SelectKBest fit inside each fold; within-subject
  tertiles computed per subject; cross-subject tertiles global (acceptable: labels
  derived from y only, applied before CV — note this is a labeling choice, not a
  feature leak).
- **Honesty:** all scripts report raw metrics; chance lines noted (R²≤0, 3-class
  0.333). Within-subject n≈30/subject is small — flagged in spec.
- **YAGNI:** band-power features only; no engineered/raw; no control epochs.
```