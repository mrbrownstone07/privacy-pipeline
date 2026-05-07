from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ClassMetrics:
    """Per-class precision / recall / F1 for one classifier."""
    name     : str
    precision: float
    recall   : float
    f1       : float
    support  : int


@dataclass
class ClassifierResult:
    """Cross-validated classification result for one model."""
    classes         : list[ClassMetrics]
    accuracy        : float
    macro_f1        : float
    macro_precision : float
    macro_recall    : float

    def __repr__(self) -> str:
        return (
            f"ClassifierResult(accuracy={self.accuracy:.4f}, "
            f"macro_f1={self.macro_f1:.4f}, "
            f"classes={len(self.classes)})"
        )


@dataclass
class AttackResult:
    """Attribute-inference attack metrics for one sensitive class (or averaged)."""
    adv_accuracy    : float
    random_baseline : float
    privacy_gain    : float   # baseline − adv_acc; > 0 is private
    norm_gain       : float   # (adv_acc − baseline) / (1 − baseline)
    adv_f1          : float
    adv_auc         : float

    @property
    def is_private(self) -> bool:
        return self.privacy_gain > 0

    def __repr__(self) -> str:
        status = "private" if self.is_private else "leaked"
        return (
            f"AttackResult(adv_auc={self.adv_auc:.4f}, "
            f"priv_gain={self.privacy_gain:+.4f}, "
            f"status={status!r})"
        )

    def to_frame(self) -> pd.DataFrame:
        """Render as a single-row DataFrame for Jupyter display."""
        return pd.DataFrame([{
            "Adv Accuracy"   : self.adv_accuracy,
            "Random Baseline": self.random_baseline,
            "Privacy Gain"   : self.privacy_gain,
            "Norm Gain"      : self.norm_gain,
            "Adv F1"         : self.adv_f1,
            "Adv AUC"        : self.adv_auc,
        }], index=["Logistic Regression (LR)"])


# ── Default classifier suite ──────────────────────────────────────────────────

_DEFAULT_CLASSIFIERS: dict[str, object] = {
    "Logistic Regression": LogisticRegression(max_iter=5000, solver="saga", random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM"                : SVC(kernel="rbf", probability=True, random_state=42),
    "KNN"                : KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"      : DecisionTreeClassifier(random_state=42),
}


# ── Public API ────────────────────────────────────────────────────────────────

def run_classification(
    embedding   : np.ndarray,
    y           : np.ndarray,
    target_names: list[str] | None = None,
    models      : dict | None      = None,
    cv          : int              = 10,
) -> dict[str, ClassifierResult]:
    """10-fold cross-validated classification. Returns one ClassifierResult per model."""
    if target_names is not None:
        target_names = list(target_names)
    models  = models or _DEFAULT_CLASSIFIERS
    results = {}
    for name, clf in models.items():
        y_pred = cross_val_predict(clf, embedding, y, cv=cv)
        rep    = classification_report(y, y_pred, target_names=target_names,
                                       output_dict=True, zero_division=0)
        results[name] = ClassifierResult(
            classes=[
                ClassMetrics(
                    name     = cls,
                    precision= round(rep[cls]["precision"], 4),
                    recall   = round(rep[cls]["recall"],    4),
                    f1       = round(rep[cls]["f1-score"],  4),
                    support  = int(rep[cls]["support"]),
                )
                for cls in (target_names or sorted({str(c) for c in y}))
            ],
            accuracy       = round(rep["accuracy"],                     4),
            macro_f1       = round(rep["macro avg"]["f1-score"],        4),
            macro_precision= round(rep["macro avg"]["precision"],       4),
            macro_recall   = round(rep["macro avg"]["recall"],          4),
        )
    return results


def display_classification_results(
    results     : dict[str, ClassifierResult],
    show_per_class: bool = False,
) -> None:
    """Render ClassifierResult dict as styled DataFrames in Jupyter."""
    def _style(df: pd.DataFrame, metric_cols: list[str]):
        return (
            df.style.format({c: "{:.3f}" for c in metric_cols})
            .highlight_max(subset=metric_cols, props="font-weight: bold;")
            .set_properties(**{"text-align": "center", "font-family": "Arial",
                               "font-size": "12px"})
            .set_table_styles([
                {"selector": "thead th",
                 "props": [("font-weight", "bold"), ("text-align", "center")]},
                {"selector": "tbody td", "props": [("padding", "4px 8px")]},
            ])
        )

    if show_per_class:
        rows = [
            {"Model": m, "Class": cm.name, "Precision": cm.precision,
             "Recall": cm.recall, "F1": cm.f1, "Support": cm.support}
            for m, r in results.items() for cm in r.classes
        ]
        display(pd.DataFrame(rows).style
                .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"})
                .set_properties(**{"text-align": "center"})
                .set_caption("Per-class metrics"))

    summary = pd.DataFrame([
        {"Model": m, "Accuracy": r.accuracy, "Macro Precision": r.macro_precision,
         "Macro Recall": r.macro_recall, "Macro F1": r.macro_f1}
        for m, r in results.items()
    ]).sort_values("Macro F1", ascending=False)
    display(_style(summary, ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"])
            .set_caption("Macro summary — ranked by Macro F1"))


def run_cv_inference_attack(
    X             : np.ndarray,
    y             : np.ndarray,
    sensitive_class,
    cv            : int  = 10,
    random_state  : int  = 42,
    stage_name    : str  = "",
    verbose       : bool = True,
) -> AttackResult:
    """
    Attribute inference attack via stratified k-fold CV.

    Adversary: LogisticRegression predicting binary membership in sensitive_class.
    Returns an AttackResult with accuracy, AUC, privacy gain, and normalized gain.
    """
    y_adv    = (y == sensitive_class).astype(int)
    pos_rate = float(y_adv.mean())
    baseline = max(pos_rate, 1.0 - pos_rate)

    splitter  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    adversary = LogisticRegression(max_iter=7000, solver="saga",
                                   random_state=random_state, class_weight="balanced")

    y_pred  = cross_val_predict(adversary, X, y_adv, cv=splitter, method="predict")
    y_proba = cross_val_predict(adversary, X, y_adv, cv=splitter,
                                method="predict_proba")[:, 1]

    adv_acc  = float(accuracy_score(y_adv, y_pred))
    adv_f1   = float(f1_score(y_adv, y_pred, zero_division=0))
    adv_auc  = float(roc_auc_score(y_adv, y_proba))
    priv_gain = baseline - adv_acc
    norm_gain = (adv_acc - baseline) / (1.0 - baseline + 1e-10)

    result = AttackResult(
        adv_accuracy   = round(adv_acc,   4),
        random_baseline= round(baseline,  4),
        privacy_gain   = round(priv_gain, 4),
        norm_gain      = round(norm_gain, 4),
        adv_f1         = round(adv_f1,    4),
        adv_auc        = round(adv_auc,   4),
    )

    if verbose:
        header = f"Inference Attack — {stage_name}" if stage_name else "Inference Attack"
        print(f"\n{'='*62}\n  {header}\n{'='*62}")
        print(f"  Sensitive class : {sensitive_class}")
        print(f"  Class balance   : {pos_rate:.1%} pos / {1-pos_rate:.1%} neg")
        print(f"  Random baseline : {baseline:.4f}\n")
        status = "✓ private" if result.is_private else "✗ leaked"
        print(f"  {'Adversary':<26} {'Acc':>6}  {'Gain':>7}  {'NormG':>6}  "
              f"{'AUC':>6}  Status")
        print(f"  {'-'*62}")
        print(f"  {'Logistic Regression (LR)':<26} {adv_acc:>6.4f} "
              f"{priv_gain:>+7.4f} {norm_gain:>6.4f} {adv_auc:>6.4f}  {status}")
        print()

    return result


def run_attack_all_classes(
    X           : np.ndarray,
    y           : np.ndarray,
    target_names: list[str],
    cv          : int  = 10,
    random_state: int  = 42,
    verbose     : bool = False,
) -> tuple[AttackResult, list[AttackResult]]:
    """
    Run inference attack for every class; return (class-averaged AttackResult,
    per-class list of AttackResults).
    """
    per_class: list[AttackResult] = []
    for cls_idx, cls_name in enumerate(target_names):
        atk = run_cv_inference_attack(
            X, y, sensitive_class=cls_idx,
            cv=cv, random_state=random_state, verbose=verbose,
        )
        per_class.append(atk)
        if verbose:
            print(f"--- Attack results for class '{cls_name}' ---")
            print(atk)

    if not per_class:
        raise ValueError("No per-class results produced.")

    def _nanmean(vals: list[float]) -> float:
        return float(np.nanmean(vals))

    avg = AttackResult(
        adv_accuracy   = _nanmean([r.adv_accuracy    for r in per_class]),
        random_baseline= _nanmean([r.random_baseline for r in per_class]),
        privacy_gain   = _nanmean([r.privacy_gain    for r in per_class]),
        norm_gain      = _nanmean([r.norm_gain       for r in per_class]),
        adv_f1         = _nanmean([r.adv_f1          for r in per_class]),
        adv_auc        = _nanmean([r.adv_auc         for r in per_class]),
    )
    return avg, per_class
