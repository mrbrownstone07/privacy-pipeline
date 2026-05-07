from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    file_path: str = "features_raw_0_overlap.csv"
    meta_cols: list[str] = field(default_factory=list)
    label_col: str = "label"
    feature_groups: list[str] | None = None  # None = all columns


@dataclass
class GraphConfig:
    n_neighbors: int = 27
    normalized: bool = True


@dataclass
class EmbeddingConfig:
    n_components: int = 4


@dataclass
class NoiseConfig:
    mechanism: str = (
        "spectral_gap"  # spectral_gap | resolvent_guided | ppsp | embedding
    )
    epsilons: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    )
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    n_splits: int = 10
    random_state: int = 42


@dataclass
class OutputConfig:
    results_dir: str = "."
    figures_dir: str = "figures"
    save_csv: bool = True
    save_figures: bool = True
    dpi: int = 150


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _build_section(cls, raw: dict):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        warnings.warn(f"Unrecognized keys in {cls.__name__}: {unknown}")
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path = "experiment.yaml") -> ExperimentConfig:
    """Load YAML config. Unknown keys warn instead of crashing."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = ExperimentConfig()
    sections = {
        "data": (DataConfig, "data"),
        "graph": (GraphConfig, "graph"),
        "embedding": (EmbeddingConfig, "embedding"),
        "noise": (NoiseConfig, "noise"),
        "evaluation": (EvaluationConfig, "evaluation"),
        "output": (OutputConfig, "output"),
    }
    for key, (cls, attr) in sections.items():
        if key in raw:
            setattr(cfg, attr, _build_section(cls, raw[key]))
    return cfg
