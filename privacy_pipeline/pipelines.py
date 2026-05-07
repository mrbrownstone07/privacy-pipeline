"""
Alternative noise-injection pipelines for scientific comparison.

All three pipelines use the **same** ε sweep, same snr_target, same
classifiers, and same inference-attack mechanism.  Only the stage at
which noise is injected differs:

    DPLaplacianEigenmaps          — noise on the graph Laplacian L  (existing)
    FeatureSpaceNoisePipeline     — noise on raw features X before graph build
    EmbeddingSpaceNoisePipeline   — noise on the clean embedding V after eigmap
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph

from .graph import EmbeddingResult
from .noise import NoiseMetadata


# ── Metadata types ────────────────────────────────────────────────────────────

@dataclass
class FeatureNoiseMetadata(NoiseMetadata):
    """Metadata for feature-space Laplace perturbation."""
    type       : str         = "feature_space"
    scales     : list[float] = field(default_factory=list)   # per-feature noise std
    snr_target : float       = 0.0
    n_features : int         = 0


@dataclass
class EmbeddingNoiseMetadata(NoiseMetadata):
    """Metadata for embedding-space Laplace perturbation."""
    type        : str         = "embedding_space"
    scales      : list[float] = field(default_factory=list)  # per-component noise std
    snr_target  : float       = 0.0
    n_components: int         = 0


# ── Shared graph/eigmap helper ─────────────────────────────────────────────────

def _laplacian_eigenmaps(
    X          : np.ndarray,
    n_neighbors: int,
    n_components: int,
    normalized : bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, sp.spmatrix]:
    """
    Build k-NN graph → Laplacian → eigenvectors.

    Returns (eigvals, eigvecs, eigvals_full, fiedler_gap, L).
    """
    A = kneighbors_graph(X, n_neighbors=n_neighbors,
                          mode="distance", include_self=False, n_jobs=-1)
    A = 0.5 * (A + A.T)
    L = laplacian(A, normed=normalized)
    ev_raw, V_raw = eigsh(L, k=n_components + 1, sigma=0)
    idx        = np.argsort(ev_raw)
    eigvals_full = ev_raw[idx]
    eigvals    = eigvals_full[1:]
    eigvecs    = V_raw[:, idx][:, 1:]
    fiedler    = float(abs(eigvals_full[1] - eigvals_full[0]))
    return eigvals, eigvecs, eigvals_full, fiedler, L


# ── Pipeline 1: Feature-space noise ──────────────────────────────────────────

class FeatureSpaceNoisePipeline:
    """
    Noise pipeline 1 — perturb raw features, then run Laplacian Eigenmaps.

    Workflow::

        X_scaled  ──(Laplace noise)──►  X_noisy
                                          │
                                     k-NN graph
                                          │
                                       Laplacian
                                          │
                                    eigenvectors  ──►  EmbeddingResult

    The clean embedding (from original X) is also computed once so that
    fiedler_gap_clean and the reference embedding are available for comparison.

    Noise scale
    -----------
    Per-feature independent Laplace noise:
        scale_j = std(X[:, j]) / (snr_target × ε)

    This gives SNR = 1 / snr_target, identical in spirit to SpectralGapNoise
    (which anchors noise to the eigenvector signal amplitude 1/√n).
    """

    def __init__(
        self,
        epsilon     : float,
        n_neighbors : int,
        n_components: int,
        normalized  : bool  = True,
        snr_target  : float = 0.625,
        random_state        = None,
    ):
        self.epsilon      = epsilon
        self.n_neighbors  = n_neighbors
        self.n_components = n_components
        self.normalized   = normalized
        self.snr_target   = snr_target
        self.rng          = np.random.default_rng(random_state)

    def fit_transform(self, X: np.ndarray) -> EmbeddingResult:
        t0 = time.perf_counter()
        n, d = X.shape

        # ── Clean reference embedding ────────────────────────────────────
        eigvals_c, eigvecs_c, _, fiedler_clean, L_clean = \
            _laplacian_eigenmaps(X, self.n_neighbors, self.n_components, self.normalized)

        # ── Add noise to features ────────────────────────────────────────
        col_std = X.std(axis=0)
        scales  = col_std / (self.snr_target * self.epsilon)
        # Vectorised: standard Laplace samples scaled per-column
        X_noisy = X + self.rng.laplace(0, 1, size=(n, d)) * scales

        # ── Laplacian Eigenmaps on noisy features ────────────────────────
        eigvals_n, eigvecs_n, _, fiedler_noisy, _ = \
            _laplacian_eigenmaps(X_noisy, self.n_neighbors, self.n_components, self.normalized)

        meta = FeatureNoiseMetadata(
            scales           = scales.tolist(),
            snr_target       = self.snr_target,
            n_features       = d,
            fiedler_gap_clean= fiedler_clean,
            fiedler_gap_noisy= fiedler_noisy,
            fiedler_gap_delta= fiedler_noisy - fiedler_clean,
            fiedler_gap_ratio= (fiedler_noisy / fiedler_clean
                                if fiedler_clean > 0 else float("inf")),
        )
        print(f"[FeatureNoise ε={self.epsilon}] "
              f"fiedler {fiedler_clean:.4e}→{fiedler_noisy:.4e} "
              f"| {time.perf_counter()-t0:.2f}s")

        return EmbeddingResult(
            embedding_clean    = eigvecs_c,
            embedding_noisy    = eigvecs_n,
            embedding_projector= eigvecs_n,          # no projector in this pipeline
            eigenvalues        = eigvals_c,
            eigenvalues_noisy  = eigvals_n,
            delta_vs           = np.zeros_like(eigvecs_c),
            fiedler_gap_clean  = fiedler_clean,
            fiedler_gap_noisy  = fiedler_noisy,
            L                  = L_clean,
            noise_metadata     = meta,
        )


# ── Pipeline 2: Embedding-space noise ────────────────────────────────────────

class EmbeddingSpaceNoisePipeline:
    """
    Noise pipeline 2 — run Laplacian Eigenmaps on clean features, then
    perturb the embedding directly.

    Workflow::

        X_scaled  ──►  k-NN graph  ──►  Laplacian  ──►  eigenvectors V
                                                              │
                                                       (Laplace noise)
                                                              │
                                                          V_noisy  ──►  EmbeddingResult

    Because the Laplacian is not modified, fiedler_gap_noisy = fiedler_gap_clean.

    Noise scale
    -----------
    Per-component independent Laplace noise:
        scale_i = std(V[:, i]) / (snr_target × ε)

    Eigenvector entries are O(1/√n); std per column ≈ 1/√n.
    This matches the SNR principle of SpectralGapNoise.
    """

    def __init__(
        self,
        epsilon     : float,
        n_neighbors : int,
        n_components: int,
        normalized  : bool  = True,
        snr_target  : float = 0.625,
        random_state        = None,
    ):
        self.epsilon      = epsilon
        self.n_neighbors  = n_neighbors
        self.n_components = n_components
        self.normalized   = normalized
        self.snr_target   = snr_target
        self.rng          = np.random.default_rng(random_state)

    def fit_transform(self, X: np.ndarray) -> EmbeddingResult:
        t0 = time.perf_counter()

        # ── Laplacian Eigenmaps on clean features ────────────────────────
        eigvals, eigvecs, _, fiedler_clean, L = \
            _laplacian_eigenmaps(X, self.n_neighbors, self.n_components, self.normalized)

        # ── Add noise to embedding ───────────────────────────────────────
        col_std = eigvecs.std(axis=0)
        scales  = col_std / (self.snr_target * self.epsilon)
        noise   = self.rng.laplace(0, 1, size=eigvecs.shape) * scales
        V_noisy = eigvecs + noise

        meta = EmbeddingNoiseMetadata(
            scales           = scales.tolist(),
            snr_target       = self.snr_target,
            n_components     = self.n_components,
            fiedler_gap_clean= fiedler_clean,
            fiedler_gap_noisy= fiedler_clean,   # L is unchanged
            fiedler_gap_delta= 0.0,
            fiedler_gap_ratio= 1.0,
        )
        print(f"[EmbeddingNoise ε={self.epsilon}] "
              f"fiedler {fiedler_clean:.4e} (L unchanged) "
              f"| {time.perf_counter()-t0:.2f}s")

        return EmbeddingResult(
            embedding_clean    = eigvecs,
            embedding_noisy    = V_noisy,
            embedding_projector= V_noisy,
            eigenvalues        = eigvals,
            eigenvalues_noisy  = eigvals.copy(),  # eigenvalues unchanged
            delta_vs           = noise,            # direct noise term
            fiedler_gap_clean  = fiedler_clean,
            fiedler_gap_noisy  = fiedler_clean,
            L                  = L,
            noise_metadata     = meta,
        )
