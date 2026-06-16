# scripts/06_search_features_loso.py
"""Compare engineered EEG feature families with LOSO cross-validation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.engineered_features import (
    build_feature_family_matrix,
    make_selected_feature_classifiers,
)
from src.evaluate import run_loso


FAMILIES = [
    "bandpower",
    "time",
    "hjorth",
    "temporal_bandpower",
    "narrow_bandpower",
    "engineered",
]
K_CANDIDATES = [20, 40, 80, 160, "all"]


def _k_value(candidate, n_features):
    if candidate == "all":
        return "all"
    return min(int(candidate), n_features)


def main():
    data = np.load(OUTPUT_DIR / "epochs.npz", allow_pickle=True)
    X_ep, y, subjects = data["X"], data["y"], data["subjects"]
    print(f"Epochs: {X_ep.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Subjects: {len(np.unique(subjects))}  (chance = 50%)\n")

    rows = []
    best = None
    for family in FAMILIES:
        F, names = build_feature_family_matrix(X_ep, family=family)
        print(f"Family {family}: {F.shape[1]} features")
        seen_k = set()
        for candidate in K_CANDIDATES:
            k = _k_value(candidate, F.shape[1])
            if k in seen_k:
                continue
            seen_k.add(k)
            for model_name, clf in make_selected_feature_classifiers(k=k).items():
                res = run_loso(F, y, subjects, clf)
                row = {
                    "family": family,
                    "n_features": F.shape[1],
                    "k": k,
                    "model": model_name,
                    "accuracy": round(res["accuracy"], 3),
                    "macro_f1": round(res["macro_f1"], 3),
                    "roc_auc": round(res["roc_auc"], 3),
                }
                rows.append(row)
                print(f"  k={str(k):>3s} {model_name:22s} "
                      f"acc={res['accuracy']:.3f} f1={res['macro_f1']:.3f} "
                      f"auc={res['roc_auc']:.3f}")
                if best is None or res["accuracy"] > best[0]["accuracy"]:
                    best = (row, res)
        print()

    out = OUTPUT_DIR / "feature_search_results.csv"
    results = pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1", "roc_auc"], ascending=False)
    results.to_csv(out, index=False)

    best_row, best_res = best
    per_subject = pd.DataFrame({
        "subject": list(best_res["per_subject"].keys()),
        "accuracy": list(best_res["per_subject"].values()),
    })
    per_subject.to_csv(OUTPUT_DIR / "feature_search_best_subjects.csv", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "feature_search_best.npz",
        labels=np.array(best_res["labels"], dtype=object),
        confusion_matrix=best_res["confusion_matrix"],
    )
    print(f"Saved: {out}")
    print("Best:", best_row)
    print("Best confusion matrix:")
    print(best_res["confusion_matrix"])


if __name__ == "__main__":
    main()
