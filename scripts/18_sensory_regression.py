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
