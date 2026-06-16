import numpy as np

from src.engineered_features import (
    build_feature_family_matrix,
    feature_family_names,
    hjorth_features,
    make_selected_feature_classifiers,
    time_domain_features,
)
from src.evaluate import run_loso


def test_time_domain_feature_names_match_values():
    epoch = np.arange(2 * 10, dtype=float).reshape(2, 10)
    values = time_domain_features(epoch)
    names = feature_family_names("time", n_channels=2)

    assert values.shape == (len(names),)
    assert "mean_ch00" in names
    assert "line_length_ch01" in names
    assert np.isfinite(values).all()


def test_hjorth_features_are_finite_for_flat_channel():
    epoch = np.ones((2, 10), dtype=float)
    values = hjorth_features(epoch)

    assert values.shape == (2 * 3,)
    assert np.isfinite(values).all()
    assert np.allclose(values, 0.0)


def test_build_feature_family_matrix_combined_names_align():
    X = np.random.RandomState(0).randn(5, 2, 30)
    F, names = build_feature_family_matrix(X, family="engineered")

    assert F.shape[0] == X.shape[0]
    assert F.shape[1] == len(names)
    assert F.shape[1] > 2 * 3
    assert np.isfinite(F).all()


def test_selected_feature_classifier_runs_loso():
    rng = np.random.RandomState(1)
    subjects = np.repeat(["S1", "S2", "S3", "S4"], 8)
    y = np.tile(np.array(["Arabica", "Robusta"] * 4), 4)
    X = rng.randn(len(y), 2, 30)
    X[y == "Arabica", 0, :10] += 1.5
    F, _ = build_feature_family_matrix(X, family="time")

    clf = make_selected_feature_classifiers(k=4)["selected_logreg"]
    res = run_loso(F, y, subjects, clf)

    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["confusion_matrix"].shape == (2, 2)


def test_make_selected_feature_classifiers_accepts_class_weight():
    import numpy as np
    from src.engineered_features import make_selected_feature_classifiers
    clfs = make_selected_feature_classifiers(k=10, class_weight="balanced")
    assert set(clfs) == {"selected_logreg", "selected_linear_svm",
                         "selected_random_forest"}
    rng = np.random.RandomState(0)
    X = rng.randn(40, 30); y = np.array(["A"] * 13 + ["B"] * 27)
    clfs["selected_logreg"].fit(X, y)          # must fit without error
