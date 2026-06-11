"""Leave-One-Subject-Out cross-validation and metrics."""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)


def make_classifiers():
    """Return dict of named sklearn pipelines (scaler + classifier)."""
    return {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000)),
        "svm_rbf": make_pipeline(StandardScaler(),
                                 SVC(kernel="rbf", probability=True)),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, random_state=0)),
    }


def run_loso(X, y, subjects, clf, positive_label="Arabica"):
    """Leave-One-Subject-Out CV. Returns aggregated metrics.

    X : (n_samples, n_features); y : labels; subjects : group ids.
    clf : an unfitted sklearn estimator/pipeline (cloned per fold).
    """
    from sklearn.base import clone
    X = np.asarray(X)
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    logo = LeaveOneGroupOut()

    all_true, all_pred, all_score = [], [], []
    per_subject = {}
    for tr, te in logo.split(X, y, groups=subjects):
        model = clone(clf)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        per_subject[str(subjects[te][0])] = accuracy_score(y[te], pred)
        all_true.extend(y[te])
        all_pred.extend(pred)
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            pi = classes.index(positive_label)
            all_score.extend(model.predict_proba(X[te])[:, pi])
        else:
            all_score.extend([np.nan] * len(te))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    labels_sorted = sorted(set(y))
    result = {
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro"),
        "confusion_matrix": confusion_matrix(all_true, all_pred,
                                             labels=labels_sorted),
        "labels": labels_sorted,
        "per_subject": per_subject,
    }
    scores = np.array(all_score, dtype=float)
    if not np.any(np.isnan(scores)):
        y_bin = (all_true == positive_label).astype(int)
        result["roc_auc"] = roc_auc_score(y_bin, scores)
    else:
        result["roc_auc"] = float("nan")
    return result
