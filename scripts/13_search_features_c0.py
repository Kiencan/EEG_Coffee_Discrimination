# scripts/13_search_features_c0.py
"""Experiment B: compare engineered feature families (LOSO) for Control vs Coffee."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.engineered_features import (build_feature_family_matrix,
                                     make_selected_feature_classifiers)
from src.evaluate import run_loso

FAMILIES = ["bandpower", "time", "hjorth", "temporal_bandpower",
            "narrow_bandpower", "engineered"]
K_CANDIDATES = [20, 40, 80, "all"]


def _k_value(candidate, n_features):
    if candidate == "all":
        return "all"
    return min(int(candidate), n_features)


def main():
    data = np.load(OUTPUT_DIR / "epochs_c0_vs_coffee.npz", allow_pickle=True)
    X_ep, y, subjects = data["X"], data["y"], data["subjects"]
    print(f"Epochs: {X_ep.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Subjects: {len(np.unique(subjects))}  (balanced chance = 0.5)\n")

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
            for model_name, clf in make_selected_feature_classifiers(
                    k=k, class_weight="balanced").items():
                res = run_loso(F, y, subjects, clf, positive_label="Coffee")
                auc = res["roc_auc"]
                row = {
                    "family": family,
                    "n_features": F.shape[1],
                    "k": k,
                    "model": model_name,
                    "balanced_acc": round(res["balanced_accuracy"], 3),
                    "macro_f1": round(res["macro_f1"], 3),
                    "roc_auc": round(auc, 3),
                }
                rows.append(row)
                print(f"  k={str(k):>3s} {model_name:24s} "
                      f"bal_acc={res['balanced_accuracy']:.3f} "
                      f"f1={res['macro_f1']:.3f} auc={auc:.3f}")
                key = (res["balanced_accuracy"],
                       0.0 if np.isnan(auc) else auc)
                if best is None or key > best[2]:
                    best = (row, res, key)
        print()

    results = pd.DataFrame(rows).sort_values(
        ["balanced_acc", "roc_auc", "macro_f1"], ascending=False)
    out = OUTPUT_DIR / "c0_vs_coffee_results.csv"
    results.to_csv(out, index=False)

    best_row, best_res, _ = best
    cm = best_res["confusion_matrix"]
    labels = best_res["labels"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(cm, cmap="Greens")
    ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels)
    ax[0].set_yticks(range(len(labels))); ax[0].set_yticklabels(labels)
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
    ax[0].set_title(f"Best: {best_row['family']}/{best_row['model']}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax[0].text(j, i, cm[i, j], ha="center", va="center")
    subs = list(best_res["per_subject"].keys())
    accs = [best_res["per_subject"][s] for s in subs]
    ax[1].bar(range(len(subs)), accs)
    ax[1].axhline(0.5, color="r", ls="--", label="chance")
    ax[1].set_xticks(range(len(subs))); ax[1].set_xticklabels(subs, rotation=90)
    ax[1].set_ylabel("Accuracy"); ax[1].set_title("Per-subject accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "c0_vs_coffee_best.png", dpi=120)

    print(f"Saved: {out} and outputs/c0_vs_coffee_best.png")
    print("Best:", best_row)
    print("Top 5:")
    print(results.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
