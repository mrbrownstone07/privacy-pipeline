import pandas as pd
from IPython.display import display

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def run_classification(embedding, y, target_names=None, models=None):
    """
    embedding: (n_samples, k) dense
    y: labels
    """
    if target_names is not None:
        target_names = list(target_names)

    if models is None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier

        models = {
            "Logistic Regression": LogisticRegression(max_iter=5000, solver="saga", random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
            "KNN":                 KNeighborsClassifier(n_neighbors=5),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
        }

    X_class = embedding

    results = {}
    for name, clf in models.items():
        y_pred = cross_val_predict(clf, X_class, y, cv=10)
        report = classification_report(y, y_pred, target_names=target_names, output_dict=True, zero_division=0)

        results[name] = {
            "classes": [
                {
                    "name":      cls,
                    "precision": round(report[cls]["precision"], 4),
                    "recall":    round(report[cls]["recall"], 4),
                    "f1":        round(report[cls]["f1-score"], 4),
                    "support":   int(report[cls]["support"]),
                }
                for cls in (target_names or sorted(set(str(c) for c in y)))
            ],
            "accuracy":        round(report["accuracy"], 4),
            "macro_f1":        round(report["macro avg"]["f1-score"], 4),
            "macro_precision": round(report["macro avg"]["precision"], 4),
            "macro_recall":    round(report["macro avg"]["recall"], 4),
        }

    return results


def display_classification_results(results, display_summary=True):
    """
    Displays classification results in clean, research-level tables:
    1. Per-class metrics for each model
    2. Macro-averaged summary across all models, ranked by Macro F1
    """

    # ── 1. Per-class table ───────────────────────────────────────────────
    rows = []
    for model, data in results.items():
        for cls in data["classes"]:
            rows.append({
                "Model": model,
                "Class": cls["name"],
                "Precision": cls["precision"],
                "Recall": cls["recall"],
                "F1": cls["f1"],
                "Support": cls["support"],
            })
    df_class = pd.DataFrame(rows)

    # ── 2. Macro summary table ───────────────────────────────────────────
    summary_rows = []
    for model, data in results.items():
        summary_rows.append({
            "Model": model,
            "Accuracy": data["accuracy"],
            "Macro Precision": data["macro_precision"],
            "Macro Recall": data["macro_recall"],
            "Macro F1": data["macro_f1"],
        })
    df_summary = pd.DataFrame(summary_rows).sort_values("Macro F1", ascending=False)

    # ── Styling function ────────────────────────────────────────────────
    def style_table(df, metric_cols):
        return (
            df.style
            .format({c: "{:.3f}" for c in metric_cols})
            .highlight_max(subset=metric_cols, props="font-weight: bold;")
            .set_properties(**{"text-align": "center", "font-family": "Arial", "font-size": "12px"})
            .set_table_styles([
                {"selector": "thead th", "props": [("font-weight", "bold"), ("text-align", "center")]},
                {"selector": "tbody td", "props": [("padding", "4px 8px")]},
            ])
        )

    # ── Display ─────────────────────────────────────────────────────────

    if not display_summary:
        display(df_class.style.format({
            "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"
        }).set_properties(**{"text-align": "center", "font-family": "Arial"}).set_caption("Per-class metrics"))

    display(style_table(df_summary, ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]).set_caption("Macro summary — ranked by Macro F1"))


def run_cv_inference_attack(
    X,
    y,
    sensitive_class,
    cv          : int  = 10,
    random_state: int  = 42,
    stage_name  : str  = "",
    verbose     : bool = True,
) -> pd.DataFrame:
    """
    Attribute Inference Attack evaluated with stratified k-fold CV.

    The adversary is given the embedding X and tries to predict binary
    membership in `sensitive_class`.  Three metrics are reported:

        Adversary Accuracy  — raw accuracy of the attack model
        Random Baseline     — majority-class rate (zero-effort adversary)
        Privacy Gain        — baseline − adv_acc
                              > 0  : adversary WORSE than random  ✓ private
                              < 0  : adversary BETTER than random ✗ leaked
        Normalized Gain     — (adv_acc − baseline) / (1 − baseline)
                              0.0  : no advantage above random
                              1.0  : perfect attack
        AUC                 — AUROC of the adversary (threshold-free)

    Fixes vs original
    -----------------
    - Removed class_weight='balanced': this artificially boosts adversary
      recall on minority class, making attack look stronger than it is.
      An honest adversary optimises for accuracy, not recall.
    - Replaced plain cv=10 with StratifiedKFold to guarantee class
      balance in every fold, especially important when sensitive_class
      is a minority.
    - Added AUC and normalized gain for richer reporting.
    - Added stage_name for easy comparison across pipeline stages.
    """

    # ── 1. Binary target ──────────────────────────────────────────────
    y_adv    = (y == sensitive_class).astype(int)
    pos_rate = float(y_adv.mean())
    baseline = max(pos_rate, 1.0 - pos_rate)   # majority-class accuracy

    # ── 2. Adversary models ───────────────────────────────────────────
    # No class_weight — honest adversary maximises accuracy, not recall
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    adversaries = {
        "Logistic Regression (LR)": LogisticRegression(
            max_iter=7000, solver="saga", random_state=random_state, class_weight="balanced",
        ),
    }

    # ── 3. Evaluate each adversary ────────────────────────────────────
    rows = {}
    for name, clf in adversaries.items():

        # Hard predictions for accuracy
        y_pred  = cross_val_predict(
            clf, X, y_adv, cv=cv_splitter, method="predict"
        )

        # Soft predictions for AUC
        y_proba = cross_val_predict(
            clf, X, y_adv, cv=cv_splitter, method="predict_proba"
        )[:, 1]

        adv_acc  = float(accuracy_score(y_adv, y_pred))
        adv_f1   = float(f1_score(y_adv, y_pred, zero_division=0))
        adv_auc  = float(roc_auc_score(y_adv, y_proba))

        priv_gain  = baseline - adv_acc
        # Normalized: how much of the exploitable gap does adversary capture?
        # 0 = no advantage, 1 = perfect attack
        norm_gain  = (adv_acc - baseline) / (1.0 - baseline + 1e-10)

        rows[name] = {
            "Adv Accuracy"   : round(adv_acc,   4),
            "Random Baseline": round(baseline,   4),
            "Privacy Gain"   : round(priv_gain,  4),   # + good, - bad
            "Norm Gain"      : round(norm_gain,  4),   # lower is better
            "Adv F1"         : round(adv_f1,     4),
            "Adv AUC"        : round(adv_auc,    4),
        }

    # ── 4. Display ────────────────────────────────────────────────────
    df = pd.DataFrame(rows).T

    if verbose:
        header = f"Inference Attack — {stage_name}" if stage_name else "Inference Attack"
        print(f"\n{'='*62}")
        print(f"  {header}")
        print(f"{'='*62}")
        print(f"  Sensitive class : {sensitive_class}")
        print(f"  Class balance   : {pos_rate:.1%} positive  /  {1-pos_rate:.1%} negative")
        print(f"  Random baseline : {baseline:.4f}")
        print()

        col_w = 26
        print(f"  {'Adversary':<{col_w}} {'Acc':>6}  {'Gain':>7}  {'NormG':>6}  {'AUC':>6}  {'Status'}")
        print(f"  {'-'*62}")
        for name, r in rows.items():
            status = "✓ private" if r["Privacy Gain"] > 0 else "✗ leaked"
            print(
                f"  {name:<{col_w}}"
                f" {r['Adv Accuracy']:>6.4f}"
                f" {r['Privacy Gain']:>+7.4f}"
                f" {r['Norm Gain']:>6.4f}"
                f" {r['Adv AUC']:>6.4f}"
                f"  {status}"
            )
        print()

    return df
