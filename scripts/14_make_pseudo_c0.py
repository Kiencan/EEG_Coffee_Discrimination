# scripts/14_make_pseudo_c0.py
"""Create class-wise pseudo-subject blocks from Control-vs-Coffee epochs."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.pseudo_subjects import (make_classwise_pseudo_subjects,
                                 pseudo_subject_report)


def main():
    data = np.load(OUTPUT_DIR / "epochs_c0_vs_coffee.npz", allow_pickle=True)
    X, y, subjects = data["X"], data["y"], data["subjects"]
    pseudo, real, pseudo_labels, block_index = make_classwise_pseudo_subjects(
        y, subjects, block_size=5)

    out = OUTPUT_DIR / "epochs_pseudo_c0.npz"
    np.savez_compressed(out, X=X, y=y, real_subjects=real,
                        pseudo_subjects=pseudo, pseudo_labels=pseudo_labels,
                        block_index=block_index)
    report = pseudo_subject_report(real, pseudo, pseudo_labels, block_index)
    report_out = OUTPUT_DIR / "pseudo_c0_report.csv"
    report.to_csv(report_out, index=False)
    print(f"Saved: {out}")
    print(f"Saved: {report_out}")
    print(f"Pseudo-subjects: {report['pseudo_subject'].nunique()}  "
          f"(real subjects: {report['real_subject'].nunique()})")
    print("Pseudo-subjects per (real_subject, label):")
    print(report.groupby(["real_subject", "label"])["pseudo_subject"].nunique())
    print("\nEpochs per pseudo-subject (describe):")
    print(report["n_epochs"].describe())


if __name__ == "__main__":
    main()
