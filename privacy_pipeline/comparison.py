"""
Three-way comparison experiment:

    Pipeline A — Laplacian-space noise  (existing DPLaplacianEigenmaps)
    Pipeline B — Feature-space noise    (FeatureSpaceNoisePipeline)
    Pipeline C — Embedding-space noise  (EmbeddingSpaceNoisePipeline)

Everything that can be shared IS shared: same ε values, same snr_target,
same classifier suite, same inference-attack mechanism and CV splits.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .evaluate import AttackResult, run_attack_all_classes, run_classification
from .experiment import BaselineRecord
from .features import Dataset, load_dataset
from .graph import DPLaplacianEigenmaps
from .noise import build_noise_mechanism
from .pipelines import EmbeddingSpaceNoisePipeline, FeatureSpaceNoisePipeline
from .features import preprocess_features as _preprocess


# ── Typed result types ────────────────────────────────────────────────────────

@dataclass
class PipelineMetrics:
    """Classification + privacy metrics for one pipeline at one ε."""
    macro_f1       : float
    adv_auc        : float
    adv_accuracy   : float
    privacy_gain   : float
    random_baseline: float
    fiedler_clean  : float
    fiedler_noisy  : float

    def __repr__(self) -> str:
        return (
            f"PipelineMetrics(f1={self.macro_f1:.4f}, "
            f"adv_auc={self.adv_auc:.4f}, "
            f"priv_gain={self.privacy_gain:+.4f})"
        )


@dataclass
class ComparisonRecord:
    """All three pipelines' metrics at a single ε."""
    epsilon   : float
    laplacian : PipelineMetrics
    feature   : PipelineMetrics
    embedding : PipelineMetrics

    def __repr__(self) -> str:
        return (
            f"ComparisonRecord(ε={self.epsilon}, "
            f"lap_f1={self.laplacian.macro_f1:.4f}, "
            f"feat_f1={self.feature.macro_f1:.4f}, "
            f"emb_f1={self.embedding.macro_f1:.4f})"
        )


@dataclass
class ComparisonResults:
    """Full output of ComparisonExperiment.run()."""
    records         : list[ComparisonRecord] = field(default_factory=list)
    baseline_records: list[BaselineRecord]   = field(default_factory=list)

    # ── DataFrame views ───────────────────────────────────────────────────────

    @property
    def long_df(self) -> pd.DataFrame:
        """Long format — one row per (ε, pipeline), ideal for grouped plots."""
        rows = []
        for rec in self.records:
            for pipeline, m in [
                ("laplacian", rec.laplacian),
                ("feature",   rec.feature),
                ("embedding", rec.embedding),
            ]:
                rows.append({
                    "epsilon"        : rec.epsilon,
                    "pipeline"       : pipeline,
                    "macro_f1"       : m.macro_f1,
                    "adv_auc"        : m.adv_auc,
                    "adv_accuracy"   : m.adv_accuracy,
                    "privacy_gain"   : m.privacy_gain,
                    "random_baseline": m.random_baseline,
                    "fiedler_clean"  : m.fiedler_clean,
                    "fiedler_noisy"  : m.fiedler_noisy,
                })
        return pd.DataFrame(rows)

    @property
    def wide_df(self) -> pd.DataFrame:
        """Wide format — one row per ε, metric columns prefixed by pipeline name."""
        rows = []
        for rec in self.records:
            row = {"epsilon": rec.epsilon}
            for pipeline, m in [
                ("lap",  rec.laplacian),
                ("feat", rec.feature),
                ("emb",  rec.embedding),
            ]:
                row.update({
                    f"{pipeline}_f1"       : m.macro_f1,
                    f"{pipeline}_auc"      : m.adv_auc,
                    f"{pipeline}_priv_gain": m.privacy_gain,
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        if not self.records:
            return "ComparisonResults(empty)"
        eps = [r.epsilon for r in self.records]
        return f"ComparisonResults(epsilons={eps})"


# ── Visualization ─────────────────────────────────────────────────────────────

_PIPELINE_STYLE: dict[str, dict] = {
    "laplacian": {"color": "#5B7CB8", "label": "Laplacian-space noise", "marker": "o"},
    "feature"  : {"color": "#E07B54", "label": "Feature-space noise",   "marker": "s"},
    "embedding": {"color": "#2D7A2D", "label": "Embedding-space noise",  "marker": "^"},
}


def plot_pipeline_comparison(
    results  : ComparisonResults,
    figsize  : tuple       = (14, 5),
    save_path: str | None  = None,
) -> plt.Figure:
    """
    Two-panel line plot:
      Left  — Macro F1 vs ε (utility)
      Right — Adversary AUC vs ε (privacy leakage)

    Each panel has one line per pipeline so the three approaches can be
    directly compared across the epsilon sweep.
    """
    df = results.long_df
    epsilons = sorted(df["epsilon"].unique())

    fig, (ax_f1, ax_auc) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("white")

    for ax in (ax_f1, ax_auc):
        ax.set_facecolor("#F7F9FC")
        ax.grid(True, color="white", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xscale("log")
        ax.set_xlabel("Privacy budget ε (log scale)", fontsize=11)

    for pipeline, style in _PIPELINE_STYLE.items():
        sub  = df[df["pipeline"] == pipeline].sort_values("epsilon")
        eps  = sub["epsilon"].values
        f1s  = sub["macro_f1"].values
        aucs = sub["adv_auc"].values

        ax_f1.plot(eps, f1s,  color=style["color"], linewidth=2, alpha=0.85,
                   marker=style["marker"], markersize=7, label=style["label"])
        ax_auc.plot(eps, aucs, color=style["color"], linewidth=2, alpha=0.85,
                    marker=style["marker"], markersize=7, label=style["label"])

    # Baseline reference lines
    for br in results.baseline_records:
        ax_f1.axhline(br.macro_f1, linestyle=":", linewidth=1.2, alpha=0.5,
                      color="#888", label=f"{br.stage} F1={br.macro_f1:.3f}")
        ax_auc.axhline(br.adv_auc, linestyle=":", linewidth=1.2, alpha=0.5,
                       color="#888", label=f"{br.stage} AUC={br.adv_auc:.3f}")

    ax_f1.set_ylabel("Macro F1 (↑ = more useful)", fontsize=11)
    ax_f1.set_title("Utility — Macro F1 across ε", fontsize=13, fontweight="bold")
    ax_f1.legend(fontsize=9, loc="lower right")

    ax_auc.set_ylabel("Adversary AUC (↓ = more private)", fontsize=11)
    ax_auc.set_title("Privacy — Adversary AUC across ε", fontsize=13, fontweight="bold")
    ax_auc.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison_tradeoff(
    results  : ComparisonResults,
    figsize  : tuple       = (10, 7.5),
    save_path: str | None  = None,
) -> plt.Figure:
    """
    Privacy–utility scatter with one trajectory per pipeline.

    Each trajectory is a sequence of (adv_auc, macro_f1) points, one per ε,
    colored by ε on a shared log-normalised colormap.
    """
    df       = results.long_df
    eps_all  = np.array(sorted(df["epsilon"].unique()))
    norm     = mcolors.LogNorm(vmin=eps_all.min(), vmax=eps_all.max())
    cmap     = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F9FC")
    ax.grid(True, color="white", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for pipeline, style in _PIPELINE_STYLE.items():
        sub      = df[df["pipeline"] == pipeline].sort_values("epsilon")
        eps_vals = sub["epsilon"].values
        aucs     = sub["adv_auc"].values
        f1s      = sub["macro_f1"].values

        ax.plot(aucs, f1s, color=style["color"], linewidth=1.8, alpha=0.6,
                linestyle="-", zorder=2, label=style["label"])
        ax.scatter(aucs, f1s, c=eps_vals, cmap=cmap, norm=norm,
                   s=130, zorder=3, edgecolors=style["color"], linewidths=1.5,
                   marker=style["marker"])

        # Label first and last point
        for idx in (0, -1):
            ax.annotate(f"ε={eps_vals[idx]:g}",
                        xy=(aucs[idx], f1s[idx]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=style["color"], fontweight="bold")

    # Shared colorbar for ε
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Privacy budget ε")

    ax.set_xlabel("Adversary AUC (↓ = more private)", fontsize=12)
    ax.set_ylabel("Macro F1 (↑ = more useful)", fontsize=12)
    ax.set_title("Privacy–Utility Tradeoff — All Pipelines", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Comparison experiment ─────────────────────────────────────────────────────

class ComparisonExperiment:
    """
    Runs all three noise-injection pipelines across the epsilon sweep defined
    in ExperimentConfig and returns a ComparisonResults object.

    Shared across all three pipelines
    ----------------------------------
    - ε values              (cfg.noise.epsilons)
    - snr_target            (cfg.noise.params["snr_target"], default 0.625)
    - n_neighbors, n_components, normalized
    - Classifier suite      (run_classification with cv=cfg.evaluation.n_splits)
    - Inference attack      (run_attack_all_classes with same cv + random_state)

    What differs
    ------------
    - Stage at which noise is injected (feature / Laplacian / embedding space)
    - Noise mechanism for the Laplacian pipeline follows cfg.noise.mechanism;
      feature and embedding pipelines always use calibrated Laplace noise
      with the same snr_target and ε.
    """

    def __init__(self, cfg: ExperimentConfig | None = None):
        self.cfg = cfg or ExperimentConfig()

    def load_data(self) -> Dataset:
        dc = self.cfg.data
        ds = load_dataset(
            file_path     = dc.file_path,
            meta_cols     = dc.meta_cols or None,
            label_col     = dc.label_col,
            feature_groups= dc.feature_groups,
        )
        return Dataset(
            X           = _preprocess(ds.X),
            y           = ds.y,
            target_names= ds.target_names,
        )

    def run(
        self,
        X_scaled    : np.ndarray,
        y           : np.ndarray,
        target_names: np.ndarray,
    ) -> ComparisonResults:
        cfg  = self.cfg
        gc   = cfg.graph
        ec   = cfg.embedding
        nc   = cfg.noise
        evc  = cfg.evaluation
        oc   = cfg.output

        # snr_target shared across all three pipelines
        snr_target = float(nc.params.get("snr_target", 0.625))

        records         : list[ComparisonRecord] = []
        baseline_records: list[BaselineRecord]   = []

        # ── Baselines (unperturbed) ───────────────────────────────────────
        from .graph import DPLaplacianEigenmaps as _DP
        _clean_model = _DP(n_neighbors=gc.n_neighbors,
                           n_components=ec.n_components,
                           normalized=gc.normalized)
        _clean = _clean_model.fit_transform(X_scaled)

        for stage, X_stage in {
            "Raw Features"   : X_scaled,
            "Clean Embedding": _clean.embedding_clean,
        }.items():
            clf = run_classification(X_stage, y, target_names, cv=evc.n_splits)
            atk, _ = run_attack_all_classes(X_stage, y, list(target_names),
                                             cv=evc.n_splits,
                                             random_state=evc.random_state)
            baseline_records.append(BaselineRecord(
                stage       = stage,
                macro_f1    = clf["Random Forest"].macro_f1,
                adv_auc     = atk.adv_auc,
                adv_accuracy= atk.adv_accuracy,
                privacy_gain= atk.privacy_gain,
            ))

        # ── Epsilon sweep ─────────────────────────────────────────────────
        for eps in nc.epsilons:
            print(f"\n── ε = {eps} ─────────────────────────────")

            # Pipeline A: Laplacian-space noise (existing mechanism from config)
            mech = build_noise_mechanism(
                name        = nc.mechanism,
                epsilon     = eps,
                y           = y,
                n_train     = len(y),
                random_state= evc.random_state,
                **nc.params,
            )
            res_L = DPLaplacianEigenmaps(
                n_neighbors   = gc.n_neighbors,
                n_components  = ec.n_components,
                noise_mechanism= mech,
                normalized    = gc.normalized,
            ).fit_transform(X_scaled)
            metrics_L = self._evaluate(res_L.embedding_noisy, y, target_names,
                                        evc, res_L.fiedler_gap_clean, res_L.fiedler_gap_noisy)

            # Pipeline B: Feature-space noise
            res_F = FeatureSpaceNoisePipeline(
                epsilon     = eps,
                n_neighbors = gc.n_neighbors,
                n_components= ec.n_components,
                normalized  = gc.normalized,
                snr_target  = snr_target,
                random_state= evc.random_state,
            ).fit_transform(X_scaled)
            metrics_F = self._evaluate(res_F.embedding_noisy, y, target_names,
                                        evc, res_F.fiedler_gap_clean, res_F.fiedler_gap_noisy)

            # Pipeline C: Embedding-space noise
            res_E = EmbeddingSpaceNoisePipeline(
                epsilon     = eps,
                n_neighbors = gc.n_neighbors,
                n_components= ec.n_components,
                normalized  = gc.normalized,
                snr_target  = snr_target,
                random_state= evc.random_state,
            ).fit_transform(X_scaled)
            metrics_E = self._evaluate(res_E.embedding_noisy, y, target_names,
                                        evc, res_E.fiedler_gap_clean, res_E.fiedler_gap_noisy)

            records.append(ComparisonRecord(
                epsilon  = eps,
                laplacian= metrics_L,
                feature  = metrics_F,
                embedding= metrics_E,
            ))

            print(f"  lap  → F1={metrics_L.macro_f1:.4f}  AUC={metrics_L.adv_auc:.4f}")
            print(f"  feat → F1={metrics_F.macro_f1:.4f}  AUC={metrics_F.adv_auc:.4f}")
            print(f"  emb  → F1={metrics_E.macro_f1:.4f}  AUC={metrics_E.adv_auc:.4f}")

        results = ComparisonResults(records=records, baseline_records=baseline_records)

        if oc.save_csv:
            from pathlib import Path
            res_dir = Path(oc.results_dir)
            res_dir.mkdir(parents=True, exist_ok=True)
            results.long_df.to_csv(res_dir / "comparison_long.csv",  index=False)
            results.wide_df.to_csv(res_dir / "comparison_wide.csv", index=False)

        if oc.save_figures:
            from pathlib import Path
            fig_dir = Path(oc.results_dir) / oc.figures_dir
            fig_dir.mkdir(parents=True, exist_ok=True)
            plot_pipeline_comparison(results).savefig(
                fig_dir / "comparison_f1_auc.png", dpi=oc.dpi, bbox_inches="tight")
            plot_comparison_tradeoff(results).savefig(
                fig_dir / "comparison_tradeoff.png", dpi=oc.dpi, bbox_inches="tight")

        return results

    @staticmethod
    def _evaluate(
        X_emb       : np.ndarray,
        y           : np.ndarray,
        target_names: np.ndarray,
        evc,
        fiedler_clean: float,
        fiedler_noisy: float,
    ) -> PipelineMetrics:
        clf = run_classification(X_emb, y, target_names, cv=evc.n_splits)
        atk, _ = run_attack_all_classes(X_emb, y, list(target_names),
                                         cv=evc.n_splits,
                                         random_state=evc.random_state)
        return PipelineMetrics(
            macro_f1       = clf["Random Forest"].macro_f1,
            adv_auc        = atk.adv_auc,
            adv_accuracy   = atk.adv_accuracy,
            privacy_gain   = atk.privacy_gain,
            random_baseline= atk.random_baseline,
            fiedler_clean  = fiedler_clean,
            fiedler_noisy  = fiedler_noisy,
        )
