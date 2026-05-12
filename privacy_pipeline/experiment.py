from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from .config import ExperimentConfig, load_config
from .evaluate import AttackResult, run_attack_all_classes, run_classification
from .features import Dataset, load_dataset, preprocess_features
from .graph import DPLaplacianEigenmaps, EmbeddingResult
from .noise import build_noise_mechanism
from .visualize import (plot_fiedler_evolution, plot_privacy_utility_tradeoff,
                         visualize_embedding_shift, visualize_graph_perturbation,
                         _reduce_2d)


def _fmt(x, spec: str = ".4f") -> str:
    """Format a scalar, returning 'N/A' for None/NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return format(x, spec)


# ── Typed record types ────────────────────────────────────────────────────────

@dataclass
class EpsilonRecord:
    """Metrics for one epsilon in the sweep."""
    epsilon        : float
    macro_f1       : float
    adv_auc        : float
    adv_accuracy   : float
    privacy_gain   : float
    random_baseline: float
    noise_scale    : float
    spectral_gap   : float
    noise_to_gap   : float
    fiedler_clean  : float
    fiedler_noisy  : float
    fiedler_ratio  : float


@dataclass
class FiedlerRecord:
    """Fiedler gap values for one epsilon."""
    epsilon      : float
    fiedler_clean: float
    fiedler_noisy: float
    fiedler_ratio: float
    fiedler_delta: float


@dataclass
class BaselineRecord:
    """Unperturbed reference stage metrics."""
    stage      : str
    macro_f1   : float
    adv_auc    : float
    adv_accuracy: float
    privacy_gain: float


@dataclass
class ExperimentResults:
    """All outputs of PrivacyExperiment.run()."""
    records         : list[EpsilonRecord]  = field(default_factory=list)
    fiedler_records : list[FiedlerRecord]  = field(default_factory=list)
    baseline_records: list[BaselineRecord] = field(default_factory=list)

    @property
    def metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(r) for r in self.records])

    @property
    def fiedler_df(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(r) for r in self.fiedler_records])

    @property
    def baseline_df(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(r) for r in self.baseline_records])

    def __repr__(self) -> str:
        return (
            f"ExperimentResults("
            f"epsilons={[r.epsilon for r in self.records]}, "
            f"f1_range=[{min(r.macro_f1 for r in self.records):.3f}, "
            f"{max(r.macro_f1 for r in self.records):.3f}])"
            if self.records else "ExperimentResults(empty)"
        )


# ── Experiment orchestrator ───────────────────────────────────────────────────

class PrivacyExperiment:
    """
    Orchestrates a full epsilon-sweep privacy experiment.

    Parameters
    ----------
    cfg : ExperimentConfig — built from load_config() or inline defaults.
    """

    def __init__(self, cfg: ExperimentConfig | None = None):
        self.cfg = cfg or ExperimentConfig()

    def load_data(self) -> Dataset:
        """Load and scale the dataset. Returns Dataset(X_scaled, y, target_names)."""
        dc = self.cfg.data
        ds = load_dataset(
            file_path     = dc.file_path,
            meta_cols     = dc.meta_cols or None,
            label_col     = dc.label_col,
            feature_groups= dc.feature_groups,
        )
        return Dataset(
            X           = preprocess_features(ds.X),
            y           = ds.y,
            target_names= ds.target_names,
        )

    def run(
        self,
        X_scaled    : np.ndarray,
        y           : np.ndarray,
        target_names: np.ndarray,
    ) -> ExperimentResults:
        """
        Full epsilon sweep: classification + inference attack + Fiedler tracking.

        Returns ExperimentResults with typed records and lazy DataFrame properties.
        """
        cfg  = self.cfg
        gc   = cfg.graph
        ec   = cfg.embedding
        nc   = cfg.noise
        evc  = cfg.evaluation
        oc   = cfg.output

        fig_dir = Path(oc.results_dir) / oc.figures_dir
        if oc.save_figures:
            fig_dir.mkdir(parents=True, exist_ok=True)

        shared_coords = None
        if oc.save_figures:
            shared_coords, _ = _reduce_2d(X_scaled, method="umap")

        records         : list[EpsilonRecord]  = []
        fiedler_records : list[FiedlerRecord]  = []
        baseline_records: list[BaselineRecord] = []

        for i, eps in enumerate(nc.epsilons):
            mechanism = build_noise_mechanism(
                name        = nc.mechanism,
                epsilon     = eps,
                y           = y,
                n_train     = len(y),
                random_state= evc.random_state,
                **nc.params,
            )
            model: EmbeddingResult = DPLaplacianEigenmaps(
                n_neighbors   = gc.n_neighbors,
                n_components  = ec.n_components,
                noise_mechanism= mechanism,
                normalized    = gc.normalized,
            ).fit_transform(X_scaled)

            # ── Baselines (computed once on the first epsilon) ────────────
            if i == 0:
                for stage, X_stage in {
                    "Raw Features"   : X_scaled,
                    "Clean Embedding": model.embedding_clean,
                }.items():
                    clf_res = run_classification(X_stage, y, target_names, cv=evc.n_splits)
                    avg_atk, _ = run_attack_all_classes(
                        X_stage, y, list(target_names),
                        cv=evc.n_splits, random_state=evc.random_state,
                    )
                    baseline_records.append(BaselineRecord(
                        stage       = stage,
                        macro_f1    = clf_res["Random Forest"].macro_f1,
                        adv_auc     = avg_atk.adv_auc,
                        adv_accuracy= avg_atk.adv_accuracy,
                        privacy_gain= avg_atk.privacy_gain,
                    ))

            # ── Fiedler tracking ──────────────────────────────────────────
            fgc = model.fiedler_gap_clean
            fgn = model.fiedler_gap_noisy
            fiedler_records.append(FiedlerRecord(
                epsilon      = eps,
                fiedler_clean= fgc,
                fiedler_noisy= fgn,
                fiedler_ratio= fgn / fgc if fgc > 0 else float("nan"),
                fiedler_delta= fgn - fgc,
            ))

            # ── Noisy embedding: classify + attack ────────────────────────
            clf_res = run_classification(
                model.embedding_noisy, y, target_names, cv=evc.n_splits)
            avg_atk, _ = run_attack_all_classes(
                model.embedding_noisy, y, list(target_names),
                cv=evc.n_splits, random_state=evc.random_state,
            )
            meta = model.noise_metadata
            records.append(EpsilonRecord(
                epsilon        = eps,
                macro_f1       = clf_res["Random Forest"].macro_f1,
                adv_auc        = avg_atk.adv_auc,
                adv_accuracy   = avg_atk.adv_accuracy,
                privacy_gain   = avg_atk.privacy_gain,
                random_baseline= avg_atk.random_baseline,
                noise_scale    = getattr(meta, "scale",        float("nan")),
                spectral_gap   = getattr(meta, "gap",          float("nan")),
                noise_to_gap   = getattr(meta, "noise_to_gap", float("nan")),
                fiedler_clean  = fgc,
                fiedler_noisy  = fgn,
                fiedler_ratio  = fiedler_records[-1].fiedler_ratio,
            ))

            # ── Per-epsilon figures ───────────────────────────────────────
            if oc.save_figures:
                import matplotlib.pyplot as plt
                fig_g, _ = visualize_graph_perturbation(
                    X_scaled, y, model,
                    target_names=list(target_names), coords=shared_coords,
                )
                fig_g.suptitle(f"k-NN graph perturbation — ε={eps}", y=1.02, fontsize=13)
                fig_g.savefig(fig_dir / f"graph_perturbation_eps_{eps}.png",
                               dpi=oc.dpi, bbox_inches="tight")
                plt.close(fig_g)

                fig_e = visualize_embedding_shift(model, y, target_names=list(target_names))
                fig_e.suptitle(f"Embedding shift — ε={eps}", y=1.02, fontsize=13)
                fig_e.savefig(fig_dir / f"embedding_shift_eps_{eps}.png",
                               dpi=oc.dpi, bbox_inches="tight")
                plt.close(fig_e)

        # ── Save CSVs ─────────────────────────────────────────────────────
        results = ExperimentResults(
            records          = records,
            fiedler_records  = fiedler_records,
            baseline_records = baseline_records,
        )
        if oc.save_csv:
            res_dir = Path(oc.results_dir)
            res_dir.mkdir(parents=True, exist_ok=True)
            results.metrics_df .to_csv(res_dir / "records_metrics.csv",  index=False)
            results.fiedler_df .to_csv(res_dir / "records_fiedler.csv",  index=False)
            results.baseline_df.to_csv(res_dir / "records_baseline.csv", index=False)

        if oc.save_figures:
            plot_privacy_utility_tradeoff(records, baseline_records=baseline_records
                ).savefig(fig_dir / "privacy_utility_tradeoff.png",
                          dpi=oc.dpi, bbox_inches="tight")
            plot_fiedler_evolution(fiedler_records).savefig(
                fig_dir / "fiedler_evolution.png", dpi=oc.dpi, bbox_inches="tight")

        return results
