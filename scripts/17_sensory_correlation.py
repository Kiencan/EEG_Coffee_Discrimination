# scripts/17_sensory_correlation.py
"""Within-subject Spearman correlation: band-power features vs ratings."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.features import build_feature_matrix, feature_names

RATINGS = ["valence", "intensity", "favourite"]


def main():
    data = np.load(OUTPUT_DIR / "epochs_sensory.npz", allow_pickle=True)
    X = data["X"]
    subjects = data["subjects"]
    F = build_feature_matrix(X)
    names = feature_names()
    rows = []
    for rating in RATINGS:
        y = data[rating].astype(float)
        acc = np.zeros(F.shape[1])
        counts = np.zeros(F.shape[1])
        for sid in np.unique(subjects):
            m = subjects == sid
            ys = y[m]
            if m.sum() < 5 or np.std(ys) == 0:
                continue
            for j in range(F.shape[1]):
                xj = F[m, j]
                if np.std(xj) == 0:
                    continue
                r, _ = spearmanr(xj, ys)
                if np.isfinite(r):
                    acc[j] += r
                    counts[j] += 1
        mean_corr = np.divide(acc, counts, out=np.zeros_like(acc),
                              where=counts > 0)
        for j in range(F.shape[1]):
            rows.append({"rating": rating, "feature": names[j],
                         "mean_within_corr": round(float(mean_corr[j]), 4),
                         "n_subjects": int(counts[j])})

    res = pd.DataFrame(rows)
    out = OUTPUT_DIR / "sensory_correlations.csv"
    res.to_csv(out, index=False)
    for rating in RATINGS:
        sub = res[res.rating == rating].copy()
        sub["abs"] = sub["mean_within_corr"].abs()
        top = sub.sort_values("abs", ascending=False).head(8)
        print(f"\nTop 8 features vs {rating} (within-subject mean Spearman):")
        print(top[["feature", "mean_within_corr", "n_subjects"]]
              .to_string(index=False))
    print("\nSummary (max |mean within-subject corr| per rating):")
    for rating in RATINGS:
        sub = res[res.rating == rating]
        print(f"  {rating}: {sub['mean_within_corr'].abs().max():.3f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
