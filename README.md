# Privacy Pipeline

**Differentially private spectral embeddings via Laplacian Eigenmaps.**

A research framework for studying the privacy-utility tradeoff when noise is injected into a graph Laplacian (or its upstream/downstream representations) before spectral dimensionality reduction. Built for bearing-fault classification on the CWRU dataset, but configurable for any tabular signal dataset.

---

## Overview

The pipeline converts raw vibration signals into spectral embeddings via **Laplacian Eigenmaps**, injects calibrated noise for differential privacy, then evaluates the resulting tradeoff between:

- **Utility** — classification accuracy (Random Forest macro F1)
- **Privacy** — resistance to a logistic-regression membership/attribute inference attack (adversary AUC)

Three noise-injection strategies are implemented for direct scientific comparison:

| Pipeline | Where noise is added |
|---|---|
| **Laplacian-space** | Perturb the graph Laplacian **L** before eigendecomposition |
| **Feature-space** | Perturb raw features **X** before graph construction |
| **Embedding-space** | Perturb eigenvectors **V** after Laplacian Eigenmaps |

All three pipelines share the same ε sweep, SNR calibration, classifier suite, and inference-attack mechanism to ensure a fair scientific comparison.

---

## Installation

Requires Python ≥ 3.14. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd privacy-pipeline
uv sync
```

Or with pip:

```bash
pip install -e .
```

Core dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `pyyaml`, `PyWavelets`, `tsfel`.

---

## Quick Start

```python
from privacy_pipeline import (
    load_config, PrivacyExperiment,
    plot_privacy_utility_tradeoff, plot_fiedler_evolution,
)

cfg = load_config("experiment.yaml")
experiment = PrivacyExperiment(cfg)
ds = experiment.load_data()
X_scaled, y, target_names = ds

results = experiment.run(X_scaled, y, target_names)
# ExperimentResults(epsilons=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0], f1_range=[0.43, 0.91])

plot_privacy_utility_tradeoff(results.records, results.baseline_records)
plot_fiedler_evolution(results.fiedler_records)
```

See [`experiment.ipynb`](experiment.ipynb) for the full interactive workflow.

---

## Configuration

All experiment parameters live in `experiment.yaml`. No code edits are needed to change datasets, mechanisms, or epsilon values.

```yaml
data:
  file_path: features_raw_0_overlap.csv
  meta_cols: [file_id, fault_size, load]
  label_col: label
  feature_groups: null          # null = all columns; or [time, frequency, wpd, acf]

graph:
  n_neighbors: 27
  normalized: true

embedding:
  n_components: 4

noise:
  mechanism: spectral_gap       # spectral_gap | resolvent_guided | ppsp | embedding
  epsilons: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
  params: {snr_target: 0.525}

evaluation:
  n_splits: 10
  random_state: 42

output:
  results_dir: results
  figures_dir: figures
  save_csv: true
  save_figures: true
  dpi: 300
```

---

## Architecture

```
privacy_pipeline/
├── config.py       — YAML config loading and dataclasses
├── features.py     — Signal segmentation, feature extraction, Dataset NamedTuple
├── graph.py        — DPLaplacianEigenmaps, EmbeddingResult, GraphDiagnostics
├── noise.py        — Four noise mechanisms + BaseNoiseMechanism interface
├── evaluate.py     — Classification, inference attack, AttackResult, ClassifierResult
├── experiment.py   — PrivacyExperiment, epsilon sweep, typed records
├── pipelines.py    — Feature-space and embedding-space noise pipelines
├── comparison.py   — Three-way ComparisonExperiment and result types
└── visualize.py    — Publication-quality plots
```

---

## Modules

### `features` — Feature Extraction and Data Loading

#### Extracting features from raw signals

The framework provides a full signal-processing pipeline for turning 1-D time-series signals into a structured feature matrix ready for Laplacian Eigenmaps.

**Step 1 — Segment your raw signal**

```python
from privacy_pipeline import segment_signal

# signal: 1-D numpy array at 12 kHz
segments = segment_signal(signal, stride=0)
# stride=0  → no overlap, step = SEG_LEN (1024 samples ≈ 85 ms)
# stride=512 → 50% overlap
# Segmentation: SEG_LEN=1024, stride=1024 (no overlap), segments=47
```

Each segment is exactly `SEG_LEN = 1024` samples. Segments that would extend past the end of the signal are discarded.

**Step 2 — Extract features**

```python
from privacy_pipeline import extract_features

metadata = [
    {"file_id": "run_01", "fault_type": "ball", "fault_size": 0.007, "load": 1}
    for _ in segments
]

df = extract_features(
    segments = segments,
    metadata = metadata,
    groups   = ["time", "frequency", "wpd", "acf"],   # default: ["wpd"]
)
# Returns a DataFrame with META_COLS + selected feature columns
```

**Available feature groups:**

| Group | Features | Count |
|---|---|---|
| `time` | `mean`, `std`, `rms`, `kurtosis`, `skewness`, `crest_factor`, `impulse_factor`, `shape_factor`, `peak_to_peak` | 9 |
| `frequency` | `dom_freq`, `freq_center`, `rms_freq`, `freq_variance`, `spec_entropy` | 5 |
| `wpd` | `wpd_{mean,std,rms,energy,kurtosis,skewness,variance}_{0..15}` — 7 statistics × 16 sub-bands (db4 wavelet, level 4) | 112 |
| `acf` | `acf_burst_spacing` — median inter-peak spacing in the autocorrelation function | 1 |

To query the column names for a given group combination without extracting features:

```python
from privacy_pipeline import feature_columns

cols = feature_columns(["time", "frequency"])
# ['mean', 'std', 'rms', ..., 'dom_freq', 'freq_center', ...]
```

**Step 3 — Save and load**

```python
df.to_csv("my_features.csv", index=False)
```

**Step 4 — Load into a `Dataset`**

```python
from privacy_pipeline import load_dataset, Dataset

ds: Dataset = load_dataset(
    file_path      = "my_features.csv",
    meta_cols      = ["file_id", "fault_type", "fault_size", "load"],
    label_col      = "fault_type",
    feature_groups = None,        # None = all non-meta columns; or e.g. ["time", "wpd"]
)
# Dataset(samples=940, features=127, classes=['ball', 'inner', 'normal', 'outer'])

X_scaled, y, target_names = ds   # NamedTuple — also unpackable as a 3-tuple
```

`load_dataset` label-encodes the `label_col` automatically. If `feature_groups` is set, only the columns matching those groups are kept — useful when you extracted all groups but want to train on a subset.

**Step 5 — Scale features**

```python
from privacy_pipeline import preprocess_features

X_scaled = preprocess_features(ds.X)   # StandardScaler fit_transform
```

`DPLaplacianEigenmaps` expects pre-scaled input — no internal scaler is applied.

**Full extraction example**

```python
import numpy as np
from privacy_pipeline import segment_signal, extract_features, load_dataset, preprocess_features

# --- per file ---
all_segments, all_meta = [], []
for file_id, (signal, fault_type, fault_size, load) in enumerate(my_data_iterator()):
    segs = segment_signal(signal, stride=0)
    all_segments.extend(segs)
    all_meta.extend([
        {"file_id": file_id, "fault_type": fault_type,
         "fault_size": fault_size, "load": load}
    ] * len(segs))

df = extract_features(all_segments, all_meta, groups=["time", "frequency", "wpd"])
df["label"] = df["fault_type"]
df.to_csv("features.csv", index=False)

# --- load for the pipeline ---
ds = load_dataset("features.csv",
                  meta_cols=["file_id", "fault_type", "fault_size", "load"],
                  label_col="label")
X_scaled = preprocess_features(ds.X)
X_scaled, y, target_names = ds.X, ds.y, ds.target_names
X_scaled = preprocess_features(X_scaled)
```

**`Dataset`** is a `NamedTuple` with fields `X`, `y`, `target_names`.

---

### `graph` — Differentially Private Laplacian Eigenmaps

```python
from privacy_pipeline import DPLaplacianEigenmaps, EmbeddingResult

model = DPLaplacianEigenmaps(
    n_neighbors=27,
    n_components=4,
    noise_mechanism=mechanism,   # any BaseNoiseMechanism; None = clean embedding
    normalized=True,
)
result: EmbeddingResult = model.fit_transform(X_scaled)
# EmbeddingResult(samples=1200, components=4, fiedler_clean=0.012800, fiedler_noisy=0.009341, mechanism='spectral_gap')
```

**`EmbeddingResult`** fields:

| Field | Type | Description |
|---|---|---|
| `embedding_clean` | `ndarray (n, k)` | Eigenvectors of the unperturbed Laplacian |
| `embedding_noisy` | `ndarray (n, k)` | Eigenvectors of the perturbed Laplacian (or post-perturbation V) |
| `embedding_projector` | `ndarray (n, k)` | First-order projector correction (perturbation theory) |
| `eigenvalues` | `ndarray (k,)` | Clean eigenvalues |
| `eigenvalues_noisy` | `ndarray (k,)` | Noisy eigenvalues |
| `delta_vs` | `ndarray (n, k)` | First-order eigenvector corrections |
| `fiedler_gap_clean` | `float` | λ₂ − λ₁ before noise |
| `fiedler_gap_noisy` | `float` | λ₂ − λ₁ after noise |
| `L` | `spmatrix` | Original graph Laplacian |
| `noise_metadata` | `NoiseMetadata` | Typed metadata from the noise mechanism |

**Graph diagnostics:**

```python
from privacy_pipeline import diagnose_knn_graph, GraphDiagnostics

diag: GraphDiagnostics = diagnose_knn_graph(A)
# GraphDiagnostics(nodes=1200, edges=16200, connected, degree=27.0/27.0/27.0)

print(diag.is_connected)    # True
print(diag.degree_mean)     # 27.0
```

---

### `noise` — Noise Mechanisms

All mechanisms implement `BaseNoiseMechanism` and return `(sp.spmatrix, NoiseMetadata)` from `.generate()`.

#### `SpectralGapNoise` — Spectral-gap–calibrated Laplace noise

Noise scale is anchored to the eigenvector signal amplitude (O(1/√n)):

```
scale = (1 / √n) / (snr_target × ε)
```

```python
from privacy_pipeline import SpectralGapNoise, build_noise_mechanism

mech = SpectralGapNoise(epsilon=0.5, snr_target=0.625)
# or via factory:
mech = build_noise_mechanism("spectral_gap", epsilon=0.5, snr_target=0.625)
```

Metadata (`SpectralGapMetadata`): `scale`, `gap`, `snr_target`, `noise_to_signal`, `noise_to_gap`, Fiedler fields.

#### `ResolventGuidedPerturbation` — Perturbation-theory–aware noise

Grounded in Greenbaum, Li & Overton (2019), Theorems 1 & 2:
- δλᵢ ≈ uᵢᵀ E uᵢ (first-order eigenvalue shift)
- ‖δuᵢ‖ ≤ ‖Sᵢ‖ · ‖E‖_F (eigenvector distortion bound)

Warns when the predicted distortion exceeds `distortion_budget`.

```python
mech = build_noise_mechanism("resolvent_guided", epsilon=0.5, distortion_budget=0.5)
```

Metadata (`ResolventGuidedMetadata`): `predicted_eigenshifts`, `actual_eigenshifts`, `shift_errors`, `first_order_valid`, `budget_exceeded`.

#### `PPSPLaplacianNoise` — MI-weighted orthogonal noise

Builds noise orthogonal to each eigenvector, weighted by mutual information with class labels. Noise is scaled to a target Frobenius norm relative to ε.

```python
mech = build_noise_mechanism("ppsp", epsilon=0.5, y=y, n_train=len(y), delta_frob=0.3)
```

Metadata (`PPSPMetadata`): `weights`, `target_frob`.

#### `EmbeddingPerturbation` — Output perturbation on V

Applies MI-weighted Laplace + Gaussian noise directly to the clean embedding (bypasses L perturbation entirely).

```python
mech = build_noise_mechanism("embedding", epsilon=0.5, y=y, n_train=len(y), snr_target=2.0)
```

Metadata (`EmbeddingPerturbationMetadata`): `col_std`, `laplace_scales`, `gaussian_sigma`, `actual_snr`, `mi_weights`, `frob_ratio`.

#### Factory

```python
from privacy_pipeline import build_noise_mechanism

mech = build_noise_mechanism(
    name="spectral_gap",   # spectral_gap | resolvent_guided | ppsp | embedding
    epsilon=1.0,
    snr_target=0.625,      # forwarded as **kwargs
)
```

---

### `evaluate` — Classification and Privacy Attack

```python
from privacy_pipeline import run_classification, run_attack_all_classes, ClassifierResult, AttackResult

# Utility: cross-validated classification
clf_res: dict[str, ClassifierResult] = run_classification(
    X_emb, y, target_names, cv=10
)
print(clf_res["Random Forest"].macro_f1)   # 0.8712
print(clf_res["SVM"].macro_f1)             # 0.8341

# Privacy: logistic regression inference attack
avg_atk, per_class_atk = run_attack_all_classes(
    X_emb, y, list(target_names), cv=10, random_state=42
)
print(avg_atk.adv_auc)        # 0.6123
print(avg_atk.random_baseline) # 0.5000  — derived from majority-class fraction
print(avg_atk.privacy_gain)   # +0.1123
print(avg_atk.is_private)     # True
```

**`ClassifierResult`** fields: `accuracy`, `macro_f1`, `macro_precision`, `macro_recall`, per-class `ClassMetrics` list.

**`AttackResult`** fields: `adv_auc`, `adv_accuracy`, `random_baseline`, `privacy_gain`, `norm_gain`, `adv_f1`. The `random_baseline` is the majority-class accuracy from the actual attack data — never a hardcoded constant.

---

### `experiment` — Epsilon Sweep

```python
from privacy_pipeline import PrivacyExperiment, ExperimentResults, EpsilonRecord

experiment = PrivacyExperiment(cfg)
ds = experiment.load_data()
X_scaled, y, target_names = ds

results: ExperimentResults = experiment.run(X_scaled, y, target_names)
# Saves CSVs and figures automatically per experiment.yaml output config

# Typed access
rec: EpsilonRecord = results.records[-1]
print(f"ε={rec.epsilon}  F1={rec.macro_f1:.4f}  AUC={rec.adv_auc:.4f}")
print(f"fiedler: {rec.fiedler_clean:.4f} → {rec.fiedler_noisy:.4f}")

# DataFrame views (lazy, built on demand)
results.metrics_df    # one row per ε
results.fiedler_df    # Fiedler gap evolution
results.baseline_df   # unperturbed reference stages
```

**`EpsilonRecord`** fields: `epsilon`, `macro_f1`, `adv_auc`, `adv_accuracy`, `privacy_gain`, `random_baseline`, `noise_scale`, `spectral_gap`, `noise_to_gap`, `fiedler_clean`, `fiedler_noisy`, `fiedler_ratio`.

**`BaselineRecord`** fields: `stage` (`"Raw Features"` or `"Clean Embedding"`), `macro_f1`, `adv_auc`, `adv_accuracy`, `privacy_gain`.

---

### `comparison` — Three-Way Pipeline Comparison

```python
from privacy_pipeline import ComparisonExperiment, ComparisonResults, ComparisonRecord

cmp_exp = ComparisonExperiment(cfg)
cmp_results: ComparisonResults = cmp_exp.run(X_scaled, y, target_names)

# Per-ε access
rec: ComparisonRecord = cmp_results.records[-1]
print(rec.laplacian.macro_f1)   # Laplacian-space pipeline
print(rec.feature.macro_f1)     # Feature-space pipeline
print(rec.embedding.macro_f1)   # Embedding-space pipeline

# DataFrame views
cmp_results.long_df   # one row per (ε, pipeline)  — ideal for grouped plots
cmp_results.wide_df   # one row per ε, columns prefixed by pipeline name
```

**`PipelineMetrics`** fields: `macro_f1`, `adv_auc`, `adv_accuracy`, `privacy_gain`, `random_baseline`, `fiedler_clean`, `fiedler_noisy`.

All three pipelines share the same `snr_target` (from `cfg.noise.params`), the same ε values, the same classifiers (10-fold CV Random Forest + SVM), and the same LR inference attack.

---

### `pipelines` — Standalone Noise Pipelines

For use outside the comparison framework:

```python
from privacy_pipeline import FeatureSpaceNoisePipeline, EmbeddingSpaceNoisePipeline

# Feature-space: noise on X, then Laplacian Eigenmaps
pipeline_f = FeatureSpaceNoisePipeline(
    epsilon=0.5, n_neighbors=27, n_components=4,
    normalized=True, snr_target=0.625,
)
result: EmbeddingResult = pipeline_f.fit_transform(X_scaled)

# Embedding-space: Laplacian Eigenmaps, then noise on V
pipeline_e = EmbeddingSpaceNoisePipeline(
    epsilon=0.5, n_neighbors=27, n_components=4,
    normalized=True, snr_target=0.625,
)
result: EmbeddingResult = pipeline_e.fit_transform(X_scaled)
```

Both return an `EmbeddingResult` with the same fields as `DPLaplacianEigenmaps`. The noise scale formula is consistent across all three spaces:

```
scale = col_std(signal) / (snr_target × ε)
```

where `signal` is the feature matrix (feature-space), the Laplacian eigenvectors (embedding-space), or implicitly the graph structure (Laplacian-space, anchored at 1/√n).

---

### `visualize` — Plots

```python
from privacy_pipeline import (
    plot_privacy_utility_tradeoff,
    plot_fiedler_evolution,
    visualize_knn_graph,
    visualize_graph_perturbation,
    visualize_embedding_shift,
    plot_pipeline_comparison,
    plot_comparison_tradeoff,
)

# Privacy-utility tradeoff curve
fig = plot_privacy_utility_tradeoff(results.records, results.baseline_records)

# Fiedler gap vs ε
fig = plot_fiedler_evolution(results.fiedler_records)

# k-NN graph colored by class
fig = visualize_knn_graph(X_scaled, y, target_names=list(target_names), n_neighbors=27)

# Comparison: F1 and AUC line plots for all three pipelines
fig = plot_pipeline_comparison(cmp_results)

# Comparison: privacy-utility scatter with ε-colored trajectories
fig = plot_comparison_tradeoff(cmp_results)
```

---

## Output Structure

Output paths are controlled entirely by the `output` block in `experiment.yaml`:

```yaml
output:
  results_dir: results      # root directory for all outputs
  figures_dir: figures      # subdirectory inside results_dir for plots
  save_csv: true
  save_figures: true
  dpi: 300
```

After running an experiment the following files are written under `{results_dir}/`:

```
{results_dir}/
├── records_metrics.csv         — per-ε classification + privacy metrics
├── records_fiedler.csv         — Fiedler gap evolution across ε
├── records_baseline.csv        — unperturbed baselines
├── comparison_long.csv         — comparison sweep (long format, one row per ε × pipeline)
├── comparison_wide.csv         — comparison sweep (wide format, one row per ε)
└── {figures_dir}/
    ├── privacy_utility_tradeoff.png
    ├── fiedler_evolution.png
    ├── graph_perturbation_eps_*.png
    ├── embedding_shift_eps_*.png
    ├── comparison_f1_auc.png
    └── comparison_tradeoff.png
```

---

## Perturbation Theory

The Laplacian-space pipeline implements first-order perturbation analysis following **Greenbaum, Li & Overton (2019)**:

- **Theorem 1** — Eigenvalue shift: δλᵢ ≈ uᵢᵀ E uᵢ
- **Theorem 2** — Eigenvector distortion: ‖δuᵢ‖ ≤ ‖Sᵢ‖ · ‖E‖_F, where Sᵢ is the resolvent

These are available as standalone functions:

```python
from privacy_pipeline import (
    eigenvalue_perturbation,
    eigenvector_perturbation,
    projector_embedding_lowrank,
)
```

---

## API Reference Summary

| Symbol | Kind | Module |
|---|---|---|
| `load_config` | function | `config` |
| `PrivacyExperiment` | class | `experiment` |
| `ExperimentResults` | dataclass | `experiment` |
| `EpsilonRecord` | dataclass | `experiment` |
| `FiedlerRecord` | dataclass | `experiment` |
| `BaselineRecord` | dataclass | `experiment` |
| `Dataset` | NamedTuple | `features` |
| `segment_signal` | function | `features` |
| `extract_features` | function | `features` |
| `feature_columns` | function | `features` |
| `load_dataset` | function | `features` |
| `preprocess_features` | function | `features` |
| `DPLaplacianEigenmaps` | class | `graph` |
| `EmbeddingResult` | dataclass | `graph` |
| `GraphDiagnostics` | dataclass | `graph` |
| `diagnose_knn_graph` | function | `graph` |
| `BaseNoiseMechanism` | class | `noise` |
| `SpectralGapNoise` | class | `noise` |
| `ResolventGuidedPerturbation` | class | `noise` |
| `PPSPLaplacianNoise` | class | `noise` |
| `EmbeddingPerturbation` | class | `noise` |
| `build_noise_mechanism` | function | `noise` |
| `run_classification` | function | `evaluate` |
| `run_attack_all_classes` | function | `evaluate` |
| `AttackResult` | dataclass | `evaluate` |
| `ClassifierResult` | dataclass | `evaluate` |
| `FeatureSpaceNoisePipeline` | class | `pipelines` |
| `EmbeddingSpaceNoisePipeline` | class | `pipelines` |
| `ComparisonExperiment` | class | `comparison` |
| `ComparisonResults` | dataclass | `comparison` |
| `ComparisonRecord` | dataclass | `comparison` |
| `PipelineMetrics` | dataclass | `comparison` |
| `plot_privacy_utility_tradeoff` | function | `visualize` |
| `plot_fiedler_evolution` | function | `visualize` |
| `plot_pipeline_comparison` | function | `visualize` |
| `plot_comparison_tradeoff` | function | `visualize` |

---

## References

- Greenbaum, A., Li, R.-C., & Overton, M. L. (2019). *First-order perturbation theory for eigenvalues and eigenvectors*. SIAM Review, 62(2), 463–482.
- Smith, W. A., & Randall, R. B. (2015). *Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study*. Mechanical Systems and Signal Processing, 64–65, 100–131.
- Belkin, M., & Niyogi, P. (2003). *Laplacian eigenmaps for dimensionality reduction and data representation*. Neural Computation, 15(6), 1373–1396.
