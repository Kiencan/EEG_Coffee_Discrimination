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
