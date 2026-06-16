# scripts/10_evaluate_averaged_blocks.py
"""Evaluate averaged class-wise blocks with real-subject LOSO."""
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
from src.evaluate import make_classifiers, run_loso
from src.features import build_feature_matrix
from src.raw_features import flatten_epochs, make_raw_pca_classifiers


def main():
    data = np.load(OUTPUT_DIR / "epochs_avg_classwise.npz", allow_pickle=True)
    X, y, groups = data["X"], data["y"], data["real_subjects"]
    print(f"Averaged epochs: {X.shape}")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Real subjects: {len(np.unique(groups))}  (chance = 50%)\n")

    rows = []
    rows.extend(_evaluate(
        build_feature_matrix(X), y, groups, "bandpower", {
            "bandpower_random_forest": make_classifiers()["random_forest"],
            "bandpower_svm_rbf": make_classifiers()["svm_rbf"],
            "bandpower_select80_linear_svm":
                make_selected_feature_classifiers(k=80)["selected_linear_svm"],
        }))
    rows.extend(_evaluate(
        flatten_epochs(X), y, groups, "raw", {
            "raw_pca_logreg": make_raw_pca_classifiers()["raw_pca_logreg"],
        }))
    F_eng, _ = build_feature_family_matrix(X, family="engineered")
    rows.extend(_evaluate(
        F_eng, y, groups, "engineered", {
            "engineered_random_forest":
                make_selected_feature_classifiers(k="all")["selected_random_forest"],
            "engineered_logreg":
                make_selected_feature_classifiers(k="all")["selected_logreg"],
        }))

    results = pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1", "roc_auc"], ascending=False)
    out = OUTPUT_DIR / "averaged_block_evaluation.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(results.to_string(index=False))


def _evaluate(F, y, groups, feature_family, models):
    rows = []
    for model_name, clf in models.items():
        res = run_loso(F, y, groups, clf)
        rows.append({
            "feature_family": feature_family,
            "model": model_name,
            "n_groups": int(len(np.unique(groups))),
            "n_samples": int(F.shape[0]),
            "n_features": int(F.shape[1]),
            "accuracy": round(res["accuracy"], 3),
            "macro_f1": round(res["macro_f1"], 3),
            "roc_auc": round(res["roc_auc"], 3),
        })
        print(f"{model_name:28s} acc={res['accuracy']:.3f} "
              f"f1={res['macro_f1']:.3f} auc={res['roc_auc']:.3f}")
    return rows


if __name__ == "__main__":
    main()
