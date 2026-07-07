"""Generate figures for the report (section 6 -- Results).

Outputs to outputs/report/:
    fig_6_1_arabica_robusta_bars.png
    fig_6_2_per_subject_rf.png
    fig_6_3_confusion_rf.png
    fig_6_4_feature_search_heatmap.png
    fig_6_5_real_vs_pseudo.png
    fig_6_6_sensory_correlation_heatmap.png

Run from project root:
    python scripts/21_results_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR, BANDS, EEG_CHANNELS
from src.evaluate import make_classifiers, run_loso
from src.features import build_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = OUTPUT_DIR / "report"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _load_epochs():
    data = np.load(OUTPUT_DIR / "epochs.npz", allow_pickle=True)
    return data["X"], data["y"], data["subjects"]


def _per_fold_metrics(F, y, subjects, clf):
    """Run LOSO and return per-fold accuracies + per-subject dict + all preds."""
    logo = LeaveOneGroupOut()
    per_fold_acc = []
    per_subject = {}
    all_true, all_pred = [], []
    for tr, te in logo.split(F, y, groups=subjects):
        model = clone(clf)
        model.fit(F[tr], y[tr])
        pred = model.predict(F[te])
        acc = accuracy_score(y[te], pred)
        per_fold_acc.append(acc)
        per_subject[str(subjects[te][0])] = acc
        all_true.extend(y[te])
        all_pred.extend(pred)
    return np.array(per_fold_acc), per_subject, np.array(all_true), np.array(all_pred)


# --------------------------------------------------------------------------
# Figure 6.1 -- bar chart, 3 classifiers on band power
# --------------------------------------------------------------------------

def fig_6_1():
    X_ep, y, subjects = _load_epochs()
    F = build_feature_matrix(X_ep)
    rows = []
    for name, clf in make_classifiers().items():
        per_fold, _, _, _ = _per_fold_metrics(F, y, subjects, clf)
        rows.append((name, per_fold.mean(), per_fold.std()))

    names = ["logreg", "svm_rbf", "random_forest"]
    pretty = {"logreg": "LogReg", "svm_rbf": "SVM (RBF)",
              "random_forest": "Random Forest"}
    means = [next(r[1] for r in rows if r[0] == n) for n in names]
    stds = [next(r[2] for r in rows if r[0] == n) for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar([pretty[n] for n in names], means, yerr=stds,
                  capsize=5, color=["#5b86b3", "#b3863b", "#6aa66a"],
                  edgecolor="black", lw=0.6)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance = 0.5")
    ax.set_ylabel("Accuracy (LOSO, 8 folds)")
    ax.set_title("Arabica vs Robusta -- band power features (160-D)")
    ax.set_ylim(0, 0.8)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.015,
                f"{m:.3f}", ha="center", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    out = FIG_DIR / "fig_6_1_arabica_robusta_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------
# Figure 6.2 -- per-subject accuracy for Random Forest
# --------------------------------------------------------------------------

def fig_6_2():
    X_ep, y, subjects = _load_epochs()
    F = build_feature_matrix(X_ep)
    clf = make_classifiers()["random_forest"]
    _, per_subject, _, _ = _per_fold_metrics(F, y, subjects, clf)

    subs = sorted(per_subject.keys())
    vals = [per_subject[s] for s in subs]
    overall = np.mean(vals)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#6aa66a" if v >= 0.5 else "#b3554b" for v in vals]
    bars = ax.bar(subs, vals, color=colors, edgecolor="black", lw=0.6)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance = 0.5")
    ax.axhline(overall, color="#222", ls=":", lw=1.2,
               label=f"overall mean = {overall:.3f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy on held-out subject")
    ax.set_title("Random Forest LOSO accuracy by held-out subject")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    out = FIG_DIR / "fig_6_2_per_subject_rf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------
# Figure 6.3 -- confusion matrix for Random Forest
# --------------------------------------------------------------------------

def fig_6_3():
    X_ep, y, subjects = _load_epochs()
    F = build_feature_matrix(X_ep)
    clf = make_classifiers()["random_forest"]
    _, _, all_true, all_pred = _per_fold_metrics(F, y, subjects, clf)
    labels = sorted(set(all_true))
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Random Forest -- confusion matrix (8 LOSO folds combined)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.0f}%)",
                    ha="center", va="center", fontsize=11, color=color)
    out = FIG_DIR / "fig_6_3_confusion_rf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------
# Figure 6.4 -- feature search heatmap (family x model, best k)
# --------------------------------------------------------------------------

def _run_feature_search():
    """Run the Arabica/Robusta feature search and return a DataFrame."""
    from src.engineered_features import (build_feature_family_matrix,
                                         make_selected_feature_classifiers)
    FAMILIES = ["bandpower", "time", "hjorth", "temporal_bandpower",
                "narrow_bandpower", "engineered"]
    K_CANDIDATES = [20, 40, 80, "all"]
    X_ep, y, subjects = _load_epochs()
    rows = []
    for family in FAMILIES:
        F, _ = build_feature_family_matrix(X_ep, family=family)
        seen_k = set()
        for cand in K_CANDIDATES:
            k = "all" if cand == "all" else min(int(cand), F.shape[1])
            if k in seen_k:
                continue
            seen_k.add(k)
            for model_name, clf in make_selected_feature_classifiers(k=k).items():
                res = run_loso(F, y, subjects, clf)
                rows.append({
                    "family": family,
                    "n_features": F.shape[1],
                    "k": k,
                    "model": model_name,
                    "balanced_acc": res["balanced_accuracy"],
                    "macro_f1": res["macro_f1"],
                    "roc_auc": res["roc_auc"],
                })
                print(f"  {family:>20s} k={str(k):>3s} {model_name:22s} "
                      f"bal_acc={res['balanced_accuracy']:.3f}")
    return pd.DataFrame(rows)


def fig_6_4():
    cache = OUTPUT_DIR / "arabica_robusta_feature_search.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"loaded cached search: {cache}")
    else:
        print("running feature search (this takes ~1-3 min)...")
        df = _run_feature_search()
        df.to_csv(cache, index=False)
        print(f"cached to {cache}")

    # For each (family, model) keep best balanced_acc across k
    best = (df.groupby(["family", "model"])["balanced_acc"]
              .max().reset_index())
    pivot = best.pivot(index="family", columns="model", values="balanced_acc")
    family_order = ["bandpower", "time", "hjorth", "narrow_bandpower",
                    "temporal_bandpower", "engineered"]
    pivot = pivot.reindex(family_order)
    model_order = ["selected_logreg", "selected_linear_svm",
                   "selected_random_forest"]
    pivot = pivot[model_order]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.40, vmax=0.60,
                   aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Balanced accuracy (best k)")
    cbar.ax.axhline(0.5, color="k", lw=1.0)

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(["LogReg", "Linear SVM", "Random Forest"])
    ax.set_yticks(range(len(family_order)))
    ax.set_yticklabels(family_order)
    ax.set_title("Feature search: balanced accuracy by family x model "
                 "(best k)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if abs(v - 0.5) > 0.07 else "black")
    out = FIG_DIR / "fig_6_4_feature_search_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------
# Figure 6.5 -- real-subject vs pseudo-block (Control vs Coffee)
# --------------------------------------------------------------------------

def fig_6_5():
    df = pd.read_csv(OUTPUT_DIR / "c0_pseudo_evaluation.csv")
    # rows have grouping in {real_subject, pseudo_block}
    cfg_order = [
        ("raw", "pca_logreg", "raw+PCA, LogReg"),
        ("raw", "pca_random_forest", "raw+PCA, RF"),
        ("engineered", "logreg", "engineered, LogReg"),
        ("engineered", "random_forest", "engineered, RF"),
        ("bandpower", "logreg", "bandpower, LogReg"),
        ("bandpower", "random_forest", "bandpower, RF"),
    ]
    labels = [c[2] for c in cfg_order]
    real_vals, pseudo_vals = [], []
    for fam, mdl, _ in cfg_order:
        r = df[(df["grouping"] == "real_subject") &
               (df["feature_family"] == fam) & (df["model"] == mdl)]
        p = df[(df["grouping"] == "pseudo_block") &
               (df["feature_family"] == fam) & (df["model"] == mdl)]
        real_vals.append(float(r["balanced_acc"].iloc[0]) if len(r) else np.nan)
        pseudo_vals.append(float(p["balanced_acc"].iloc[0]) if len(p) else np.nan)

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    b1 = ax.bar(x - w / 2, real_vals, w, label="Real subject (8 groups)",
                color="#5b86b3", edgecolor="black", lw=0.5)
    b2 = ax.bar(x + w / 2, pseudo_vals, w, label="Pseudo block (72 groups)",
                color="#b3863b", edgecolor="black", lw=0.5)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance = 0.5")
    for bars, vals in [(b1, real_vals), (b2, pseudo_vals)]:
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.30, 0.58)
    ax.set_title("Control vs Coffee -- real-subject vs pseudo-block LOSO")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    out = FIG_DIR / "fig_6_5_real_vs_pseudo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------
# Figure 6.6 -- sensory correlation heatmap (within-subject)
# --------------------------------------------------------------------------

def fig_6_6():
    df = pd.read_csv(OUTPUT_DIR / "sensory_correlations.csv")
    # Build a wide matrix: rows = 160 features, cols = 3 ratings
    ratings = ["valence", "intensity", "favourite"]
    bands = list(BANDS.keys())  # delta..gamma
    feat_order = []
    for prefix in ("abs", "rel"):
        for band in bands:
            for ch in EEG_CHANNELS:
                feat_order.append(f"{prefix}_{band}_{ch}")

    mat = np.zeros((len(feat_order), len(ratings)))
    for ri, rating in enumerate(ratings):
        sub = df[df["rating"] == rating].set_index("feature")
        for fi, feat in enumerate(feat_order):
            mat[fi, ri] = sub.loc[feat, "mean_within_corr"] if feat in sub.index else np.nan

    vmax = float(np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(5.5, 10))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean within-subject Pearson r")

    ax.set_xticks(range(len(ratings)))
    ax.set_xticklabels(ratings)
    ax.set_yticks([])

    # Group separators between abs and rel (after 80 rows), and between bands
    rows_per_band = len(EEG_CHANNELS)  # 16
    band_count = len(bands)            # 5
    rel_start = band_count * rows_per_band  # 80
    ax.axhline(rel_start - 0.5, color="black", lw=1.0)
    for k in range(1, band_count):
        ax.axhline(k * rows_per_band - 0.5, color="gray", lw=0.3, ls=":")
        ax.axhline(rel_start + k * rows_per_band - 0.5,
                   color="gray", lw=0.3, ls=":")

    # Group labels on the left (per band x abs/rel)
    for prefix_idx, prefix in enumerate(("abs", "rel")):
        for bi, band in enumerate(bands):
            y = prefix_idx * rel_start + bi * rows_per_band + rows_per_band / 2
            ax.text(-0.55, y - 0.5, f"{prefix}\n{band}",
                    ha="right", va="center", fontsize=8)

    ax.set_title("Within-subject correlation:\nband power feature x sensory rating")
    ax.set_xlim(-0.5, len(ratings) - 0.5)
    out = FIG_DIR / "fig_6_6_sensory_correlation_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# --------------------------------------------------------------------------

def main():
    fig_6_1()
    fig_6_2()
    fig_6_3()
    fig_6_4()
    fig_6_5()
    fig_6_6()


if __name__ == "__main__":
    main()
