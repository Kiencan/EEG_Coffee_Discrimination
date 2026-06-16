# Control vs Coffee Feature-comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify Control (C0) vs Coffee (C1∪C2) from 3-second EEG epochs of the 8 clean subjects, comparing raw-data and multiple engineered feature families under LOSO with balanced metrics, to find the best representation.

**Architecture:** Merge the `codex/eeg-feature-experiments` branch into `master` to reuse its `raw_features` and `engineered_features` modules. Add a control-aware epoch labeling path in `eeg_io`, a Control-vs-Coffee dataset builder, balanced metrics in `evaluate.run_loso`, and two experiment scripts (raw, feature-search). All evaluation is subject-independent LOSO with `class_weight='balanced'`.

**Tech Stack:** Python 3.14, NumPy 2.x, scipy, scikit-learn, matplotlib, pytest.

---

## File Structure

```
src/eeg_io.py            # + label_for_task, extract_epochs_task (control-aware)
src/evaluate.py          # + balanced_accuracy, decision_function AUC fallback
src/raw_features.py      # (from codex) + class_weight param
src/engineered_features.py # (from codex) + class_weight param
scripts/11_build_c0_vs_coffee.py  # build outputs/epochs_c0_vs_coffee.npz
scripts/12_raw_loso_c0.py         # Experiment A (raw + PCA)
scripts/13_search_features_c0.py  # Experiment B (feature-family search)
tests/test_eeg_io.py     # + label_for_task / extract_epochs_task tests
tests/test_evaluate.py   # + balanced_accuracy test
tests/test_raw_features.py        # (from codex) + class_weight test
tests/test_engineered_features.py # (from codex) + class_weight test
```

**Branch note:** work happens on `master`. Task 1 merges codex. `master` is the only writable branch here (codex is checked out in another worktree — do NOT check it out).

---

## Task 1: Merge codex feature modules into master

**Files:** repo-wide merge (brings in `src/raw_features.py`, `src/engineered_features.py`, `src/pseudo_subjects.py`, `scripts/05-10`, codex tests).

- [ ] **Step 1: Confirm clean working tree on master**

Run: `git status -sb && git branch`
Expected: on `master`, nothing uncommitted in tracked files (untracked `data/ docs/ protocol/` are fine — they were already untracked).

- [ ] **Step 2: Merge the codex branch**

Run: `git merge --no-edit codex/eeg-feature-experiments`
Expected: a merge commit, or fast-forward-style integration. Only potential conflict is `.gitignore`.

- [ ] **Step 3: If `.gitignore` conflicts, resolve by keeping both sides**

If Step 2 reports a conflict in `.gitignore`, open it, remove the `<<<<<<< ======= >>>>>>>` markers so the file contains the union of both versions' lines (all of: `__pycache__/`, `*.pyc`, `outputs/`, `.pytest_cache/`, `*.npz`, plus any line codex added). Then:
Run: `git add .gitignore && git commit --no-edit`

- [ ] **Step 4: Verify codex modules now exist on master and the suite passes**

Run: `python -m pytest -q`
Expected: ALL tests pass (master's ~20 + codex's raw/engineered/pseudo tests). If a codex test fails due to the NumPy 2.x `np.trapz` issue, note it; `engineered_features.py` already uses `np.trapezoid`, so it should pass.

- [ ] **Step 5: Sanity-check the reused factories import**

Run:
```bash
python -c "from src.raw_features import flatten_epochs, make_raw_pca_classifiers; from src.engineered_features import build_feature_family_matrix, make_selected_feature_classifiers; print('ok', list(make_raw_pca_classifiers().keys()))"
```
Expected: `ok ['raw_pca_logreg', 'raw_pca_linear_svm', 'raw_pca_random_forest']`.

No extra commit needed beyond the merge.

---

## Task 2: Control-aware epoch labeling (`eeg_io`)

**Files:**
- Modify: `src/eeg_io.py`
- Test: `tests/test_eeg_io.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_eeg_io.py`:

```python
from src.eeg_io import label_for_task, extract_epochs_task

def test_label_for_task_coffee_default():
    # default task reproduces old coffee labeling
    assert label_for_task(981) == "Arabica"
    assert label_for_task(585) == "Robusta"
    assert label_for_task(712) is None

def test_label_for_task_control_vs_coffee():
    assert label_for_task(712, task="control_vs_coffee") == "Control"   # C0
    assert label_for_task(981, task="control_vs_coffee") == "Coffee"    # C1
    assert label_for_task(585, task="control_vs_coffee") == "Coffee"    # C2
    assert label_for_task(0, task="control_vs_coffee") is None          # ISI
    assert label_for_task(1, task="control_vs_coffee") is None          # baseline

def test_extract_epochs_task_control_vs_coffee():
    from src.config import EPOCH_LEN
    n_ch = 16
    rest = np.zeros((4, n_ch))
    ctrl = np.ones((EPOCH_LEN, n_ch)) * 1.0       # C0 code 712
    ara = np.ones((EPOCH_LEN, n_ch)) * 2.0        # C1 code 981
    rob = np.ones((EPOCH_LEN, n_ch)) * 3.0        # C2 code 585
    signals = np.vstack([rest, ctrl, rest, ara, rest, rob, rest])
    codes = np.concatenate([
        np.zeros(4), np.full(EPOCH_LEN, 712), np.zeros(4),
        np.full(EPOCH_LEN, 981), np.zeros(4),
        np.full(EPOCH_LEN, 585), np.zeros(4)])
    X, y, _ = extract_epochs_task(signals, codes, task="control_vs_coffee",
                                  expected_len=EPOCH_LEN)
    assert X.shape == (3, n_ch, EPOCH_LEN)
    assert list(y) == ["Control", "Coffee", "Coffee"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_eeg_io.py -k "task" -v`
Expected: FAIL with `ImportError: cannot import name 'label_for_task'`.

- [ ] **Step 3: Implement** — in `src/eeg_io.py`, update the config import line and add the two functions; also refactor `extract_epochs` to delegate (DRY).

Change the existing import line:
```python
from src.config import (ARABICA_CODES, ROBUSTA_CODES, EEG_CHANNELS, EPOCH_LEN)
```
to:
```python
from src.config import (ARABICA_CODES, ROBUSTA_CODES, CONTROL_CODES,
                        EEG_CHANNELS, EPOCH_LEN)
```

Add after `label_for_code`:
```python
def label_for_task(code, task="coffee"):
    """Map a stimulus code to a label for a given classification task.

    task="coffee"            -> Arabica / Robusta / None (original behavior).
    task="control_vs_coffee" -> Control (C0) / Coffee (C1 or C2) / None.
    """
    if task == "coffee":
        return label_for_code(code)
    if task == "control_vs_coffee":
        if code in CONTROL_CODES:
            return "Control"
        if code in ARABICA_CODES or code in ROBUSTA_CODES:
            return "Coffee"
        return None
    raise ValueError(f"Unknown task: {task}")


def extract_epochs_task(signals, codes, task="coffee", expected_len=EPOCH_LEN):
    """Extract labeled epochs for a given task (see label_for_task)."""
    signals = np.asarray(signals)
    epochs, labels, run_codes = [], [], []
    for code, start, length in find_code_runs(codes):
        label = label_for_task(code, task=task)
        if label is None:
            continue
        if length < expected_len:
            continue
        seg = signals[start:start + expected_len]
        epochs.append(seg.T)
        labels.append(label)
        run_codes.append(code)
    if not epochs:
        return (np.empty((0, signals.shape[1], expected_len)),
                np.array([], dtype=object), np.array([], dtype=int))
    return np.stack(epochs), np.array(labels, dtype=object), np.array(run_codes)
```

Then replace the body of the existing `extract_epochs` so it delegates (keeps old behavior/tests intact). Find the existing `def extract_epochs(signals, codes, expected_len=EPOCH_LEN):` function and replace its entire body with:
```python
def extract_epochs(signals, codes, expected_len=EPOCH_LEN):
    """Extract coffee-sniff epochs (Arabica/Robusta). Thin wrapper kept for
    backward compatibility; delegates to extract_epochs_task(task="coffee")."""
    return extract_epochs_task(signals, codes, task="coffee",
                               expected_len=expected_len)
```

- [ ] **Step 4: Run to verify pass (new + existing eeg_io tests)**

Run: `python -m pytest tests/test_eeg_io.py -v`
Expected: PASS (existing extract_epochs tests + 3 new task tests).

- [ ] **Step 5: Commit**

```bash
git add src/eeg_io.py tests/test_eeg_io.py
git commit -m "feat: control-aware epoch labeling (label_for_task, extract_epochs_task)"
```

---

## Task 3: Balanced metrics + AUC fallback (`evaluate`)

**Files:**
- Modify: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_evaluate.py`:

```python
def test_run_loso_reports_balanced_accuracy():
    rng = np.random.RandomState(1)
    n_per = 30
    subjects = np.repeat(["S1", "S2", "S3"], n_per)
    # imbalanced: 10 Control, 20 Coffee per subject
    y = np.tile(np.array(["Control"] * 10 + ["Coffee"] * 20), 3)
    X = rng.randn(len(y), 6)
    X[y == "Coffee", 0] += 2.5
    clf = make_classifiers()["logreg"]
    res = run_loso(X, y, subjects, clf, positive_label="Coffee")
    assert "balanced_accuracy" in res
    assert 0.0 <= res["balanced_accuracy"] <= 1.0
    assert res["balanced_accuracy"] > 0.7
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_evaluate.py -k balanced -v`
Expected: FAIL with `KeyError: 'balanced_accuracy'`.

- [ ] **Step 3: Implement** — in `src/evaluate.py`:

Update the metrics import:
```python
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
```
to:
```python
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, confusion_matrix)
```

Replace the entire `run_loso` function with:
```python
def run_loso(X, y, subjects, clf, positive_label="Arabica"):
    """Leave-One-Subject-Out CV. Returns aggregated metrics.

    X : (n_samples, n_features); y : labels; subjects : group ids.
    clf : an unfitted sklearn estimator/pipeline (cloned per fold).
    Probability scores for AUC use predict_proba, falling back to
    decision_function (for classifiers like LinearSVC) when needed.
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
        classes = list(model.classes_)
        if positive_label in classes and hasattr(model, "predict_proba"):
            pi = classes.index(positive_label)
            all_score.extend(model.predict_proba(X[te])[:, pi])
        elif positive_label in classes and hasattr(model, "decision_function"):
            d = np.asarray(model.decision_function(X[te]), dtype=float)
            if d.ndim == 1:
                # binary decision_function is oriented toward classes[1]
                d = d if classes.index(positive_label) == 1 else -d
                all_score.extend(d)
            else:
                all_score.extend([np.nan] * len(te))
        else:
            all_score.extend([np.nan] * len(te))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    labels_sorted = sorted(set(y))
    result = {
        "accuracy": accuracy_score(all_true, all_pred),
        "balanced_accuracy": balanced_accuracy_score(all_true, all_pred),
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
```

- [ ] **Step 4: Run to verify pass (new + existing evaluate tests)**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS (existing 2 + new balanced test).

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: balanced_accuracy and decision_function AUC fallback in run_loso"
```

---

## Task 4: class_weight parameter in feature classifier factories

**Files:**
- Modify: `src/raw_features.py`, `src/engineered_features.py`
- Test: `tests/test_raw_features.py`, `tests/test_engineered_features.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_raw_features.py`:
```python
def test_make_raw_pca_classifiers_accepts_class_weight():
    import numpy as np
    from src.raw_features import make_raw_pca_classifiers
    clfs = make_raw_pca_classifiers(class_weight="balanced")
    assert set(clfs) == {"raw_pca_logreg", "raw_pca_linear_svm",
                         "raw_pca_random_forest"}
    rng = np.random.RandomState(0)
    X = rng.randn(40, 50); y = np.array(["A"] * 13 + ["B"] * 27)
    clfs["raw_pca_logreg"].fit(X, y)          # must fit without error
```

Append to `tests/test_engineered_features.py`:
```python
def test_make_selected_feature_classifiers_accepts_class_weight():
    import numpy as np
    from src.engineered_features import make_selected_feature_classifiers
    clfs = make_selected_feature_classifiers(k=10, class_weight="balanced")
    assert set(clfs) == {"selected_logreg", "selected_linear_svm",
                         "selected_random_forest"}
    rng = np.random.RandomState(0)
    X = rng.randn(40, 30); y = np.array(["A"] * 13 + ["B"] * 27)
    clfs["selected_logreg"].fit(X, y)          # must fit without error
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_raw_features.py tests/test_engineered_features.py -k class_weight -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'class_weight'`.

- [ ] **Step 3: Implement**

In `src/raw_features.py`, change the signature and the three estimators of `make_raw_pca_classifiers`:
```python
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
```

In `src/engineered_features.py`, change `make_selected_feature_classifiers`:
```python
def make_selected_feature_classifiers(k=80, class_weight=None):
    """Return sklearn pipelines with univariate selection inside each fold.

    `class_weight` is forwarded to the final estimators (e.g. "balanced").
    """
    return {
        "selected_logreg": make_pipeline(
            StandardScaler(),
            SelectKBest(score_func=f_classif, k=k),
            LogisticRegression(max_iter=2000, random_state=0,
                               class_weight=class_weight),
        ),
        "selected_linear_svm": make_pipeline(
            StandardScaler(),
            SelectKBest(score_func=f_classif, k=k),
            LinearSVC(max_iter=5000, random_state=0,
                      class_weight=class_weight),
        ),
        "selected_random_forest": make_pipeline(
            StandardScaler(),
            SelectKBest(score_func=f_classif, k=k),
            RandomForestClassifier(n_estimators=300, random_state=0,
                                   class_weight=class_weight),
        ),
    }
```

- [ ] **Step 4: Run to verify pass (new + existing module tests)**

Run: `python -m pytest tests/test_raw_features.py tests/test_engineered_features.py -v`
Expected: PASS (existing codex tests + 2 new class_weight tests).

- [ ] **Step 5: Commit**

```bash
git add src/raw_features.py src/engineered_features.py tests/test_raw_features.py tests/test_engineered_features.py
git commit -m "feat: optional class_weight in feature classifier factories"
```

---

## Task 5: Build Control-vs-Coffee dataset (`scripts/11`)

**Files:**
- Create: `scripts/11_build_c0_vs_coffee.py`

- [ ] **Step 1: Write the script**

```python
# scripts/11_build_c0_vs_coffee.py
"""Build Control (C0) vs Coffee (C1 union C2) epoch set from clean subjects."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (DATA_DIR, OUTPUT_DIR, FS, EPOCH_LEN, BANDPASS_LOW,
                        BANDPASS_HIGH, FILTER_ORDER, USE_CAR, FLAT_STD_UV,
                        EPOCH_PTP_PERCENTILE, EXCLUDE_SUBJECTS)
from src.eeg_io import load_subject, extract_epochs_task
from src.preprocess import bandpass_filter, common_average_reference
from src.quality import epoch_is_artifact


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    files = sorted(DATA_DIR.glob("P*_KT88_with_times.csv"))
    raw_epochs, raw_labels, raw_subjects = [], [], []
    for f in files:
        sig, codes, sid = load_subject(f)
        if sid in EXCLUDE_SUBJECTS:
            continue
        filt = bandpass_filter(sig.T, fs=FS, low=BANDPASS_LOW,
                               high=BANDPASS_HIGH, order=FILTER_ORDER).T
        X, y, _ = extract_epochs_task(filt, codes, task="control_vs_coffee",
                                      expected_len=EPOCH_LEN)
        for ep, lab in zip(X, y):
            if USE_CAR:
                ep = common_average_reference(ep)
            raw_epochs.append(ep)
            raw_labels.append(lab)
            raw_subjects.append(sid)

    raw_epochs = np.stack(raw_epochs)
    raw_labels = np.array(raw_labels, dtype=object)
    raw_subjects = np.array(raw_subjects, dtype=object)
    ptp = np.array([np.max(np.ptp(ep, axis=1)) for ep in raw_epochs])
    ptp_thresh = float(np.nanpercentile(ptp, EPOCH_PTP_PERCENTILE))
    keep = np.array([not epoch_is_artifact(ep, ptp_thresh, FLAT_STD_UV)
                     for ep in raw_epochs])
    X = raw_epochs[keep]
    y = raw_labels[keep]
    subjects = raw_subjects[keep]
    out = OUTPUT_DIR / "epochs_c0_vs_coffee.npz"
    np.savez_compressed(out, X=X, y=y, subjects=subjects)
    print(f"Kept {keep.sum()}/{len(keep)} epochs (ptp_thresh={ptp_thresh:.1f} uV)")
    print(f"Subjects ({len(set(subjects.tolist()))}): {sorted(set(subjects.tolist()))}")
    uniq, cnt = np.unique(y, return_counts=True)
    print("Class balance:", dict(zip(uniq.tolist(), cnt.tolist())))
    print(f"Saved: {out}  X={X.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on real data**

Run: `python scripts/11_build_c0_vs_coffee.py`
Expected: 8 subjects (P001, P014-P020), no NaN, class balance roughly `{'Control': ~120, 'Coffee': ~240}`, X shape around `(~355, 16, 300)`. Capture full stdout.

- [ ] **Step 3: Confirm no NaN in the saved set**

Run: `python -c "import numpy as np; d=np.load('outputs/epochs_c0_vs_coffee.npz', allow_pickle=True); print('any nan:', np.isnan(d['X'].astype(float)).any(), d['X'].shape, len(np.unique(d['subjects'])))"`
Expected: `any nan: False` and 8 unique subjects.

- [ ] **Step 4: Commit**

```bash
git add scripts/11_build_c0_vs_coffee.py
git commit -m "feat: Control-vs-Coffee dataset builder (clean subjects)"
```

---

## Task 6: Experiment A — raw-data LOSO (`scripts/12`)

**Files:**
- Create: `scripts/12_raw_loso_c0.py`

- [ ] **Step 1: Write the script**

```python
# scripts/12_raw_loso_c0.py
"""Experiment A: raw-data (flatten + PCA) LOSO for Control vs Coffee."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.evaluate import run_loso
from src.raw_features import flatten_epochs, make_raw_pca_classifiers


def main():
    data = np.load(OUTPUT_DIR / "epochs_c0_vs_coffee.npz", allow_pickle=True)
    X_ep, y, subjects = data["X"], data["y"], data["subjects"]
    F = flatten_epochs(X_ep)
    print(f"Raw epochs: {X_ep.shape} -> flattened: {F.shape}")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Subjects: {len(np.unique(subjects))}  (balanced chance = 0.5)\n")

    summary = []
    for name, clf in make_raw_pca_classifiers(class_weight="balanced").items():
        res = run_loso(F, y, subjects, clf, positive_label="Coffee")
        summary.append({
            "model": name,
            "balanced_acc": round(res["balanced_accuracy"], 3),
            "macro_f1": round(res["macro_f1"], 3),
            "roc_auc": round(res["roc_auc"], 3),
        })
        print(f"{name:24s} bal_acc={res['balanced_accuracy']:.3f} "
              f"f1={res['macro_f1']:.3f} auc={res['roc_auc']:.3f}")

    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "c0_raw_loso_results.csv",
                                 index=False)
    print("\nSaved: outputs/c0_raw_loso_results.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/12_raw_loso_c0.py`
Expected: prints balanced-acc/f1/auc for the 3 raw_pca models and saves
`outputs/c0_raw_loso_results.csv`. Report the actual numbers honestly.

- [ ] **Step 3: Commit**

```bash
git add scripts/12_raw_loso_c0.py
git commit -m "feat: Experiment A raw-data LOSO for Control vs Coffee"
```

---

## Task 7: Experiment B — feature-family search (`scripts/13`)

**Files:**
- Create: `scripts/13_search_features_c0.py`

- [ ] **Step 1: Write the script**

```python
# scripts/13_search_features_c0.py
"""Experiment B: compare engineered feature families (LOSO) for Control vs Coffee."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUTPUT_DIR
from src.engineered_features import (build_feature_family_matrix,
                                     make_selected_feature_classifiers)
from src.evaluate import run_loso

FAMILIES = ["bandpower", "time", "hjorth", "temporal_bandpower",
            "narrow_bandpower", "engineered"]
K_CANDIDATES = [20, 40, 80, "all"]


def _k_value(candidate, n_features):
    if candidate == "all":
        return "all"
    return min(int(candidate), n_features)


def main():
    data = np.load(OUTPUT_DIR / "epochs_c0_vs_coffee.npz", allow_pickle=True)
    X_ep, y, subjects = data["X"], data["y"], data["subjects"]
    print(f"Epochs: {X_ep.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Subjects: {len(np.unique(subjects))}  (balanced chance = 0.5)\n")

    rows = []
    best = None
    for family in FAMILIES:
        F, names = build_feature_family_matrix(X_ep, family=family)
        print(f"Family {family}: {F.shape[1]} features")
        seen_k = set()
        for candidate in K_CANDIDATES:
            k = _k_value(candidate, F.shape[1])
            if k in seen_k:
                continue
            seen_k.add(k)
            for model_name, clf in make_selected_feature_classifiers(
                    k=k, class_weight="balanced").items():
                res = run_loso(F, y, subjects, clf, positive_label="Coffee")
                auc = res["roc_auc"]
                row = {
                    "family": family,
                    "n_features": F.shape[1],
                    "k": k,
                    "model": model_name,
                    "balanced_acc": round(res["balanced_accuracy"], 3),
                    "macro_f1": round(res["macro_f1"], 3),
                    "roc_auc": round(auc, 3),
                }
                rows.append(row)
                print(f"  k={str(k):>3s} {model_name:24s} "
                      f"bal_acc={res['balanced_accuracy']:.3f} "
                      f"f1={res['macro_f1']:.3f} auc={auc:.3f}")
                key = (res["balanced_accuracy"],
                       0.0 if np.isnan(auc) else auc)
                if best is None or key > best[2]:
                    best = (row, res, key)
        print()

    results = pd.DataFrame(rows).sort_values(
        ["balanced_acc", "roc_auc", "macro_f1"], ascending=False)
    out = OUTPUT_DIR / "c0_vs_coffee_results.csv"
    results.to_csv(out, index=False)

    best_row, best_res, _ = best
    cm = best_res["confusion_matrix"]
    labels = best_res["labels"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(cm, cmap="Greens")
    ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels)
    ax[0].set_yticks(range(len(labels))); ax[0].set_yticklabels(labels)
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
    ax[0].set_title(f"Best: {best_row['family']}/{best_row['model']}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax[0].text(j, i, cm[i, j], ha="center", va="center")
    subs = list(best_res["per_subject"].keys())
    accs = [best_res["per_subject"][s] for s in subs]
    ax[1].bar(range(len(subs)), accs)
    ax[1].axhline(0.5, color="r", ls="--", label="chance")
    ax[1].set_xticks(range(len(subs))); ax[1].set_xticklabels(subs, rotation=90)
    ax[1].set_ylabel("Accuracy"); ax[1].set_title("Per-subject accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "c0_vs_coffee_best.png", dpi=120)

    print(f"Saved: {out} and outputs/c0_vs_coffee_best.png")
    print("Best:", best_row)
    print("Top 5:")
    print(results.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/13_search_features_c0.py`
Expected: per-family/per-model balanced-acc/f1/auc lines, a saved
`outputs/c0_vs_coffee_results.csv` ranked by balanced accuracy, a
`outputs/c0_vs_coffee_best.png`, and a printed "Best" feature/model + top-5
table. Report the actual best feature and its balanced accuracy honestly
(near 0.5 means no separable Control-vs-Coffee signal).

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/13_search_features_c0.py
git commit -m "feat: Experiment B feature-family search for Control vs Coffee"
```

---

## Self-Review Notes

- **Spec coverage:** merge codex (Task 1), control labeling (Task 2), balanced
  metrics + class_weight (Tasks 3-4), dataset builder for clean 8 subjects
  (Task 5), Experiment A raw (Task 6), Experiment B feature search w/ SelectKBest
  ranked by balanced accuracy (Task 7). Outputs `c0_vs_coffee_results.csv` +
  `c0_vs_coffee_best.png` per spec.
- **Imbalance:** `class_weight='balanced'` passed in Tasks 6-7; primary metric
  is `balanced_accuracy` (Task 3); chance = 0.5.
- **Backward compatibility:** `extract_epochs` now delegates to
  `extract_epochs_task(task="coffee")` — unchanged behavior; `run_loso` only adds
  keys and fills previously-nan AUC. Existing A-vs-R scripts/tests unaffected.
- **No leakage:** PCA, SelectKBest, scaler are all inside the per-fold pipeline.
- **YAGNI:** no deep learning; k fixed candidate set; no within-subject CV.
```