"""
classification_eval.py
─────────────────────────────────────────────────────────────────────────────
Robust cross-validated classification evaluation.

Fixes applied
─────────────────────────────────────────────────────────────────────────────
1. StratifiedKFold (shuffled)  – ensures balanced class representation per fold
2. Mean ± std per metric       – exposes instability instead of hiding it
3. DummyClassifier baseline    – contextualises every other result
4. Low-support class flagging  – warns when per-class metrics are unreliable
5. Balanced-accuracy + MCC     – imbalance-robust summary metrics
6. Statistical significance    – Wilcoxon test between every model and Dummy
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


LOW_SUPPORT_THRESHOLD = 30   # classes with fewer samples get a warning flag
N_SPLITS               = 10  # CV folds
RANDOM_STATE           = 42


def _default_models() -> dict:
    return {
        "Dummy (baseline)":  DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=5000, solver="saga", random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    }

def _cv_mean_std(estimator, X, y, scorer, cv) -> tuple[float, float]:
    """Return (mean, std) of a scorer across CV folds."""
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scorer)
    return float(scores.mean()), float(scores.std())


def _significance_vs_dummy(
    model_scores: np.ndarray,
    dummy_scores: np.ndarray,
) -> dict:
    """
    Paired Wilcoxon signed-rank test between model and dummy fold scores.
    Returns p-value and a human-readable verdict.
    """
    if np.allclose(model_scores, dummy_scores):
        return {"p_value": 1.0, "significant": False, "note": "identical to dummy"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = wilcoxon(model_scores, dummy_scores)

    return {
        "p_value":     round(float(p), 4),
        "significant": bool(p < 0.05),
        "note":        "better than dummy" if (p < 0.05 and model_scores.mean() > dummy_scores.mean())
                       else ("worse than dummy" if p < 0.05 else "not distinguishable from dummy"),
    }


def _per_class_stats(y_pred, y, target_names, classes) -> list[dict]:
    """Build per-class precision / recall / f1 / support with low-support flag."""
    from sklearn.metrics import classification_report
    report = classification_report(
        y, y_pred,
        labels=classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    results = []
    for cls_name in target_names:
        entry = report[cls_name]
        support = int(entry["support"])
        results.append({
            "name":         cls_name,
            "precision":    round(entry["precision"], 4),
            "recall":       round(entry["recall"],    4),
            "f1":           round(entry["f1-score"],  4),
            "support":      support,
            "low_support":  support < LOW_SUPPORT_THRESHOLD,
        })
    return results


def run_classification(
    embedding,
    y,
    target_names: Optional[list] = None,
    models:       Optional[dict] = None,
) -> dict:
    """
    Evaluate classifiers on `embedding` using stratified k-fold CV.

    Parameters
    ----------
    embedding    : array-like (n_samples, n_features)
    y            : array-like (n_samples,) — class labels
    target_names : optional list of human-readable class names
    models       : optional dict of {name: sklearn estimator};
                   defaults to five standard classifiers + a dummy baseline

    Returns
    -------
    dict keyed by model name, each containing:
        classes         – per-class metrics with low_support flag
        accuracy        – mean ± std over folds
        balanced_acc    – mean ± std (robust to imbalance)
        macro_f1        – mean ± std
        mcc             – mean ± std (Matthews Correlation Coefficient)
        vs_dummy        – Wilcoxon significance test against the dummy baseline
    """
    X = np.asarray(embedding)
    y = np.asarray(y)

    classes      = sorted(set(y))
    target_names = list(target_names) if target_names is not None else [str(c) for c in classes]
    models       = models if models is not None else _default_models()

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scorers = {
        "accuracy":     "accuracy",
        "balanced_acc": make_scorer(balanced_accuracy_score),
        "macro_f1":     make_scorer(f1_score, average="macro", zero_division=0),
        "mcc":          make_scorer(matthews_corrcoef),
    }

    dummy_clf    = models.get("Dummy (baseline)") or DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy_scores = cross_val_score(dummy_clf, X, y, cv=cv, scoring=scorers["macro_f1"])

    results = {}

    for name, clf in models.items():
        fold_scores = {
            metric: cross_val_score(clf, X, y, cv=cv, scoring=scorer)
            for metric, scorer in scorers.items()
        }
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        sig = _significance_vs_dummy(fold_scores["macro_f1"], dummy_scores)

        results[name] = {
            "classes":     _per_class_stats(y_pred, y, target_names, classes),
            "accuracy":    {"mean": round(fold_scores["accuracy"].mean(), 4), "std": round(fold_scores["accuracy"].std(), 4)},
            "balanced_acc":{"mean": round(fold_scores["balanced_acc"].mean(), 4), "std": round(fold_scores["balanced_acc"].std(), 4)},
            "macro_f1":    {"mean": round(fold_scores["macro_f1"].mean(), 4), "std": round(fold_scores["macro_f1"].std(), 4)},
            "mcc":         {"mean": round(fold_scores["mcc"].mean(), 4), "std": round(fold_scores["mcc"].std(), 4)},
            "vs_dummy":    sig,
        }

    return results
