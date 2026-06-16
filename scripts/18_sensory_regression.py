# scripts/18_sensory_regression.py
"""Predict continuous sensory ratings from band-power EEG (within + cross).

Within-subject metric = MEAN over subjects of each subject's own CV R2 and
Pearson r (computed relative to that subject's mean). This isolates genuine
trial-level prediction; a pooled R2 would be inflated by between-subject mean
differences. Cross-subject = pooled LOSO.
"""
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
    """Mean across subjects of each subject's own within-subject CV R2/Pearson."""
    r2s, rs = [], []
    n_used = 0
    for sid in np.unique(subjects):
        m = np.where(subjects == sid)[0]
        if len(m) < 10:
            continue
        preds = np.full(len(m), np.nan)
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in kf.split(m):
            model = _pipe()
            model.fit(F[m[tr]], y[m[tr]])
            preds[te] = model.predict(F[m[te]])
        r2s.append(r2_score(y[m], preds))
        if np.std(preds) > 0:
            rs.append(pearsonr(y[m], preds)[0])
        n_used += len(m)
    return float(np.mean(r2s)), float(np.mean(rs)), n_used


def _cross_subject(F, y, subjects):
    """Pooled LOSO R2 and Pearson r across all held-out subjects."""
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
    print("\nNote: within_subject R2/r are means over subjects of each subject's")
    print("own CV (relative to that subject's mean) -- not pooled.")
    print(f"Saved: {OUTPUT_DIR / 'sensory_regression.csv'}")


if __name__ == "__main__":
    main()
