# scripts/04_train_loso.py
"""Run LOSO cross-validation for each classifier and report results."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.evaluate import make_classifiers, run_loso


def main():
    data = np.load(OUTPUT_DIR / "features.npz", allow_pickle=True)
    F, y, subjects = data["F"], data["y"], data["subjects"]
    print(f"Data: {F.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Subjects: {len(np.unique(subjects))}  (chance = 50%)\n")

    summary = []
    best = None
    for name, clf in make_classifiers().items():
        res = run_loso(F, y, subjects, clf)
        summary.append({
            "model": name,
            "accuracy": round(res["accuracy"], 3),
            "macro_f1": round(res["macro_f1"], 3),
            "roc_auc": round(res["roc_auc"], 3),
        })
        print(f"{name:14s} acc={res['accuracy']:.3f} f1={res['macro_f1']:.3f} "
              f"auc={res['roc_auc']:.3f}")
        if best is None or res["accuracy"] > best[1]["accuracy"]:
            best = (name, res)

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "loso_results.csv", index=False)

    name, res = best
    cm = res["confusion_matrix"]
    labels = res["labels"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    im = ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels)
    ax[0].set_yticks(range(len(labels))); ax[0].set_yticklabels(labels)
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
    ax[0].set_title(f"Confusion ({name})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax[0].text(j, i, cm[i, j], ha="center", va="center")
    subs = list(res["per_subject"].keys())
    accs = [res["per_subject"][s] for s in subs]
    ax[1].bar(range(len(subs)), accs)
    ax[1].axhline(0.5, color="r", ls="--", label="chance")
    ax[1].set_xticks(range(len(subs))); ax[1].set_xticklabels(subs, rotation=90)
    ax[1].set_ylabel("Accuracy"); ax[1].set_title("Per-subject accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loso_figures.png", dpi=120)
    print(f"\nBest: {name}. Saved loso_results.csv and loso_figures.png")


if __name__ == "__main__":
    main()
