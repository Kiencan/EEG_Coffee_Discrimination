# scripts/09_make_averaged_classwise_blocks.py
"""Average each complete class-wise block of five epochs into one sample."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.pseudo_subjects import average_classwise_blocks, averaged_block_report


def main():
    data = np.load(OUTPUT_DIR / "epochs.npz", allow_pickle=True)
    averaged = average_classwise_blocks(
        data["X"], data["y"], data["subjects"],
        block_size=5,
        drop_incomplete=True,
    )

    out = OUTPUT_DIR / "epochs_avg_classwise.npz"
    np.savez_compressed(out, **averaged)
    report = averaged_block_report(averaged)
    report_out = OUTPUT_DIR / "averaged_block_report.csv"
    report.to_csv(report_out, index=False)

    print(f"Saved: {out}")
    print(f"Saved: {report_out}")
    print(f"Averaged samples: {averaged['X'].shape[0]}")
    print(f"X={averaged['X'].shape}")
    print(f"Classes: {dict(zip(*np.unique(averaged['y'], return_counts=True)))}")
    print(f"Real subjects: {len(np.unique(averaged['real_subjects']))}")


if __name__ == "__main__":
    main()
