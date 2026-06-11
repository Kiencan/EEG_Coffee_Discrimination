# scripts/03_features.py
"""Build the band-power feature matrix from clean epochs."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR, FS
from src.features import build_feature_matrix, feature_names


def main():
    data = np.load(OUTPUT_DIR / "epochs.npz", allow_pickle=True)
    X_ep, y, subjects = data["X"], data["y"], data["subjects"]
    F = build_feature_matrix(X_ep, fs=FS)
    out = OUTPUT_DIR / "features.npz"
    np.savez_compressed(out, F=F, y=y, subjects=subjects,
                        names=np.array(feature_names(), dtype=object))
    print(f"Feature matrix: {F.shape}  ->  {out}")


if __name__ == "__main__":
    main()
