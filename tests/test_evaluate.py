import numpy as np
from src.evaluate import run_loso, make_classifiers

def test_no_subject_leakage_in_folds():
    from sklearn.model_selection import LeaveOneGroupOut
    groups = np.repeat(["A", "B", "C", "D"], 10)
    logo = LeaveOneGroupOut()
    X = np.random.randn(40, 5)
    for tr, te in logo.split(X, np.zeros(40), groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))

def test_run_loso_returns_metrics():
    rng = np.random.RandomState(0)
    n_per = 20
    subjects = np.repeat(["S1", "S2", "S3", "S4"], n_per)
    y = np.tile(np.array(["Arabica", "Robusta"] * (n_per // 2)), 4)
    X = rng.randn(len(y), 6)
    X[y == "Arabica", 0] += 3.0
    clf = make_classifiers()["logreg"]
    res = run_loso(X, y, subjects, clf)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["accuracy"] > 0.8
    assert set(res["per_subject"].keys()) == {"S1", "S2", "S3", "S4"}
    assert res["confusion_matrix"].shape == (2, 2)
