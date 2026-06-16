import numpy as np

from src.evaluate import run_loso
from src.raw_features import flatten_epochs, make_raw_pca_classifiers


def test_flatten_epochs_preserves_channel_time_order():
    X = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    F = flatten_epochs(X)
    assert F.shape == (2, 12)
    assert np.array_equal(F[0], X[0].reshape(-1))
    assert np.array_equal(F[1], X[1].reshape(-1))


def test_raw_pca_classifier_runs_loso():
    rng = np.random.RandomState(0)
    n_per = 8
    subjects = np.repeat(["S1", "S2", "S3", "S4"], n_per)
    y = np.tile(np.array(["Arabica", "Robusta"] * (n_per // 2)), 4)
    X = rng.randn(len(y), 2, 20)
    X[y == "Arabica", 0, :5] += 2.0
    F = flatten_epochs(X)

    clf = make_raw_pca_classifiers(n_components=0.9)["raw_pca_logreg"]
    res = run_loso(F, y, subjects, clf)

    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["confusion_matrix"].shape == (2, 2)
