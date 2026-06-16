"""Raw time-series feature baselines for preprocessed EEG epochs."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def flatten_epochs(X_epochs):
    """Map (n_epochs, n_channels, n_samples) -> (n_epochs, n_raw_features)."""
    X_epochs = np.asarray(X_epochs, dtype=float)
    if X_epochs.ndim != 3:
        raise ValueError("X_epochs must have shape (n_epochs, n_channels, n_samples)")
    return X_epochs.reshape(X_epochs.shape[0], -1)


def make_raw_pca_classifiers(n_components=0.95, class_weight=None):
    """Return classifiers for flattened raw epochs with PCA inside each fold.

    PCA is part of the pipeline so LOSO evaluation fits it only on the training
    subjects in each fold. `n_components` may be a variance fraction such as
    0.95 or a fixed integer component count. `class_weight` is passed to the
    final estimators (e.g. "balanced" for imbalanced tasks).
    """
    def pca():
        return PCA(n_components=n_components, svd_solver="full")

    return {
        "raw_pca_logreg": make_pipeline(
            StandardScaler(),
            pca(),
            LogisticRegression(max_iter=2000, random_state=0,
                               class_weight=class_weight),
        ),
        "raw_pca_linear_svm": make_pipeline(
            StandardScaler(),
            pca(),
            LinearSVC(max_iter=5000, random_state=0,
                      class_weight=class_weight),
        ),
        "raw_pca_random_forest": make_pipeline(
            StandardScaler(),
            pca(),
            RandomForestClassifier(n_estimators=300, random_state=0,
                                   class_weight=class_weight),
        ),
    }
