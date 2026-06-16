# scripts/08_evaluate_classwise_pseudo.py
"""Evaluate stage-2 class-wise pseudo blocks with two grouping schemes."""
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


def _rows_for_grouping(X_ep, y, groups, grouping_name):
    rows = []

    F_band = build_feature_matrix(X_ep)
    band_models = {
        "bandpower_random_forest": make_classifiers()["random_forest"],
        "bandpower_svm_rbf": make_classifiers()["svm_rbf"],
        "bandpower_select80_linear_svm":
            make_selected_feature_classifiers(k=80)["selected_linear_svm"],
    }
    rows.extend(_evaluate_models(
        F_band, y, groups, grouping_name, "bandpower", band_models))

    F_raw = flatten_epochs(X_ep)
    raw_models = {
        "raw_pca_logreg": make_raw_pca_classifiers()["raw_pca_logreg"],
    }
    rows.extend(_evaluate_models(
        F_raw, y, groups, grouping_name, "raw", raw_models))

    F_eng, _ = build_feature_family_matrix(X_ep, family="engineered")
    engineered_models = {
        "engineered_random_forest":
            make_selected_feature_classifiers(k="all")["selected_random_forest"],
        "engineered_logreg":
            make_selected_feature_classifiers(k="all")["selected_logreg"],
    }
    rows.extend(_evaluate_models(
        F_eng, y, groups, grouping_name, "engineered", engineered_models))
    return rows


def _evaluate_models(F, y, groups, grouping_name, feature_family, models):
    rows = []
    for model_name, clf in models.items():
        res = run_loso(F, y, groups, clf)
        rows.append({
            "grouping": grouping_name,
            "feature_family": feature_family,
            "model": model_name,
            "n_groups": int(len(np.unique(groups))),
            "n_features": int(F.shape[1]),
            "accuracy": round(res["accuracy"], 3),
            "macro_f1": round(res["macro_f1"], 3),
            "roc_auc": round(res["roc_auc"], 3),
        })
        print(f"{grouping_name:14s} {model_name:28s} "
              f"acc={res['accuracy']:.3f} f1={res['macro_f1']:.3f} "
              f"auc={res['roc_auc']:.3f}")
    return rows


def main():
    data = np.load(OUTPUT_DIR / "epochs_pseudo_classwise.npz", allow_pickle=True)
    X, y = data["X"], data["y"]
    real_subjects = data["real_subjects"]
    pseudo_subjects = data["pseudo_subjects"]
    print(f"Epochs: {X.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Pseudo groups: {len(np.unique(pseudo_subjects))}")
    print(f"Real groups: {len(np.unique(real_subjects))}\n")

    rows = []
    rows.extend(_rows_for_grouping(X, y, pseudo_subjects, "pseudo_block"))
    rows.extend(_rows_for_grouping(X, y, real_subjects, "real_subject"))

    out = OUTPUT_DIR / "classwise_pseudo_evaluation.csv"
    results = pd.DataFrame(rows).sort_values(
        ["grouping", "accuracy", "macro_f1", "roc_auc"],
        ascending=[True, False, False, False],
    )
    results.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
