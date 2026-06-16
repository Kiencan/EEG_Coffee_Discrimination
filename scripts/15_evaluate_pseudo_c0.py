# scripts/15_evaluate_pseudo_c0.py
"""Evaluate Control-vs-Coffee class-wise pseudo blocks: pseudo vs real grouping."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.engineered_features import (build_feature_family_matrix,
                                     make_selected_feature_classifiers)
from src.evaluate import run_loso
from src.features import build_feature_matrix
from src.raw_features import flatten_epochs, make_raw_pca_classifiers


def _evaluate_models(F, y, groups, grouping_name, feature_family, models):
    rows = []
    for model_name, clf in models.items():
        res = run_loso(F, y, groups, clf, positive_label="Coffee")
        rows.append({
            "grouping": grouping_name,
            "feature_family": feature_family,
            "model": model_name,
            "n_groups": int(len(np.unique(groups))),
            "n_features": int(F.shape[1]),
            "balanced_acc": round(res["balanced_accuracy"], 3),
            "macro_f1": round(res["macro_f1"], 3),
            "roc_auc": round(res["roc_auc"], 3),
        })
        print(f"{grouping_name:13s} {feature_family:11s} {model_name:22s} "
              f"bal_acc={res['balanced_accuracy']:.3f} "
              f"f1={res['macro_f1']:.3f} auc={res['roc_auc']:.3f}")
    return rows


def _rows_for_grouping(X_ep, y, groups, grouping_name):
    rows = []
    F_band = build_feature_matrix(X_ep)
    band = make_selected_feature_classifiers(k="all", class_weight="balanced")
    rows.extend(_evaluate_models(
        F_band, y, groups, grouping_name, "bandpower",
        {"random_forest": band["selected_random_forest"],
         "logreg": band["selected_logreg"]}))

    F_eng, _ = build_feature_family_matrix(X_ep, family="engineered")
    eng = make_selected_feature_classifiers(k="all", class_weight="balanced")
    rows.extend(_evaluate_models(
        F_eng, y, groups, grouping_name, "engineered",
        {"random_forest": eng["selected_random_forest"],
         "logreg": eng["selected_logreg"]}))

    F_raw = flatten_epochs(X_ep)
    raw = make_raw_pca_classifiers(class_weight="balanced")
    rows.extend(_evaluate_models(
        F_raw, y, groups, grouping_name, "raw",
        {"pca_logreg": raw["raw_pca_logreg"],
         "pca_random_forest": raw["raw_pca_random_forest"]}))
    return rows


def main():
    data = np.load(OUTPUT_DIR / "epochs_pseudo_c0.npz", allow_pickle=True)
    X, y = data["X"], data["y"]
    real_subjects = data["real_subjects"]
    pseudo_subjects = data["pseudo_subjects"]
    print(f"Epochs: {X.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Pseudo groups: {len(np.unique(pseudo_subjects))}  "
          f"Real groups: {len(np.unique(real_subjects))}  (balanced chance = 0.5)\n")

    rows = []
    rows.extend(_rows_for_grouping(X, y, pseudo_subjects, "pseudo_block"))
    print()
    rows.extend(_rows_for_grouping(X, y, real_subjects, "real_subject"))

    out = OUTPUT_DIR / "c0_pseudo_evaluation.csv"
    results = pd.DataFrame(rows).sort_values(
        ["grouping", "balanced_acc", "macro_f1", "roc_auc"],
        ascending=[True, False, False, False])
    results.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
