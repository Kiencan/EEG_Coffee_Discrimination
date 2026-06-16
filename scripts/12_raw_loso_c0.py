# scripts/12_raw_loso_c0.py
"""Experiment A: raw-data (flatten + PCA) LOSO for Control vs Coffee."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.evaluate import run_loso
from src.raw_features import flatten_epochs, make_raw_pca_classifiers


def main():
    data = np.load(OUTPUT_DIR / "epochs_c0_vs_coffee.npz", allow_pickle=True)
    X_ep, y, subjects = data["X"], data["y"], data["subjects"]
    F = flatten_epochs(X_ep)
    print(f"Raw epochs: {X_ep.shape} -> flattened: {F.shape}")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Subjects: {len(np.unique(subjects))}  (balanced chance = 0.5)\n")

    summary = []
    for name, clf in make_raw_pca_classifiers(class_weight="balanced").items():
        res = run_loso(F, y, subjects, clf, positive_label="Coffee")
        summary.append({
            "model": name,
            "balanced_acc": round(res["balanced_accuracy"], 3),
            "macro_f1": round(res["macro_f1"], 3),
            "roc_auc": round(res["roc_auc"], 3),
        })
        print(f"{name:24s} bal_acc={res['balanced_accuracy']:.3f} "
              f"f1={res['macro_f1']:.3f} auc={res['roc_auc']:.3f}")

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "c0_raw_loso_results.csv",
                                 index=False)
    print("\nSaved: outputs/c0_raw_loso_results.csv")


if __name__ == "__main__":
    main()
