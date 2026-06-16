"""Utilities for class-wise pseudo-subject/block assignment."""
import numpy as np
import pandas as pd


def make_classwise_pseudo_subjects(y, subjects, block_size=5):
    """Assign class-specific pseudo-subject IDs within each real subject.

    For each real subject and label, epochs keep their existing order and are
    split into consecutive blocks of `block_size`. Partial final blocks are
    retained so artifact-rejected subjects do not lose data.
    """
    y = np.asarray(y, dtype=object)
    subjects = np.asarray(subjects, dtype=object)
    if y.shape[0] != subjects.shape[0]:
        raise ValueError("y and subjects must have the same length")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    pseudo = np.empty(y.shape[0], dtype=object)
    block_index = np.empty(y.shape[0], dtype=int)
    for subject in _ordered_unique(subjects):
        subject_mask = subjects == subject
        for label in _ordered_unique(y[subject_mask]):
            idx = np.where(subject_mask & (y == label))[0]
            for pos, epoch_idx in enumerate(idx):
                block = pos // block_size + 1
                pseudo[epoch_idx] = f"{subject}_{label}_B{block:02d}"
                block_index[epoch_idx] = block
    return pseudo, subjects.copy(), y.copy(), block_index


def pseudo_subject_report(real_subjects, pseudo_subjects, labels, block_index):
    """Summarize class-wise pseudo-subject blocks as a pandas DataFrame."""
    real_subjects = np.asarray(real_subjects, dtype=object)
    pseudo_subjects = np.asarray(pseudo_subjects, dtype=object)
    labels = np.asarray(labels, dtype=object)
    block_index = np.asarray(block_index, dtype=int)
    rows = []
    for pseudo in _ordered_unique(pseudo_subjects):
        mask = pseudo_subjects == pseudo
        unique_labels = _ordered_unique(labels[mask])
        unique_real = _ordered_unique(real_subjects[mask])
        unique_blocks = np.unique(block_index[mask])
        rows.append({
            "real_subject": unique_real[0],
            "pseudo_subject": pseudo,
            "label": unique_labels[0],
            "block_index": int(unique_blocks[0]),
            "n_epochs": int(np.sum(mask)),
        })
    return pd.DataFrame(rows)


def average_classwise_blocks(X, y, subjects, block_size=5, drop_incomplete=True):
    """Average raw epochs inside each class-wise block.

    Returns a dict containing averaged epochs plus metadata. With
    `drop_incomplete=True`, only exact `block_size` blocks are retained.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=object)
    subjects = np.asarray(subjects, dtype=object)
    if X.shape[0] != y.shape[0] or y.shape[0] != subjects.shape[0]:
        raise ValueError("X, y, and subjects must have the same first dimension")
    if X.ndim != 3:
        raise ValueError("X must have shape (n_epochs, n_channels, n_samples)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    avg_epochs = []
    avg_y = []
    avg_real = []
    avg_pseudo = []
    avg_block = []
    avg_n_epochs = []
    for subject in _ordered_unique(subjects):
        subject_mask = subjects == subject
        for label in _ordered_unique(y[subject_mask]):
            idx = np.where(subject_mask & (y == label))[0]
            for start in range(0, len(idx), block_size):
                block_idx = idx[start:start + block_size]
                if drop_incomplete and len(block_idx) != block_size:
                    continue
                block = start // block_size + 1
                avg_epochs.append(X[block_idx].mean(axis=0))
                avg_y.append(label)
                avg_real.append(subject)
                avg_pseudo.append(f"{subject}_{label}_B{block:02d}")
                avg_block.append(block)
                avg_n_epochs.append(len(block_idx))

    return {
        "X": np.asarray(avg_epochs, dtype=float),
        "y": np.asarray(avg_y, dtype=object),
        "real_subjects": np.asarray(avg_real, dtype=object),
        "pseudo_subjects": np.asarray(avg_pseudo, dtype=object),
        "block_index": np.asarray(avg_block, dtype=int),
        "n_epochs": np.asarray(avg_n_epochs, dtype=int),
    }


def averaged_block_report(averaged):
    """Summarize averaged class-wise blocks as a pandas DataFrame."""
    return pd.DataFrame({
        "real_subject": averaged["real_subjects"],
        "pseudo_subject": averaged["pseudo_subjects"],
        "label": averaged["y"],
        "block_index": averaged["block_index"],
        "n_epochs_averaged": averaged["n_epochs"],
    })


def _ordered_unique(values):
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen
