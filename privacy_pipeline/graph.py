from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph

from .noise import BaseNoiseMechanism, NoiseMetadata


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """Full output of DPLaplacianEigenmaps.fit_transform."""
    embedding_clean    : np.ndarray = field(repr=False)
    embedding_noisy    : np.ndarray = field(repr=False)
    embedding_projector: np.ndarray = field(repr=False)
    eigenvalues        : np.ndarray = field(repr=False)
    eigenvalues_noisy  : np.ndarray = field(repr=False)
    delta_vs           : np.ndarray = field(repr=False)
    fiedler_gap_clean  : float      = 0.0
    fiedler_gap_noisy  : float      = 0.0
    L                  : sp.spmatrix = field(repr=False, default=None)
    noise_metadata     : NoiseMetadata | None = None

    def __repr__(self) -> str:
        n, k  = self.embedding_clean.shape
        mech  = self.noise_metadata.type if self.noise_metadata else "none"
        return (
            f"EmbeddingResult(samples={n}, components={k}, "
            f"fiedler_clean={self.fiedler_gap_clean:.6f}, "
            f"fiedler_noisy={self.fiedler_gap_noisy:.6f}, "
            f"mechanism={mech!r})"
        )


@dataclass
class GraphDiagnostics:
    """Connectivity diagnostics for a k-NN adjacency matrix."""
    n_nodes         : int
    n_edges         : int
    n_components    : int
    is_connected    : bool
    degree_min      : float
    degree_mean     : float
    degree_max      : float
    component_sizes : list[int] = field(default_factory=list)

    def __repr__(self) -> str:
        conn = "connected" if self.is_connected else f"{self.n_components} components"
        return (
            f"GraphDiagnostics(nodes={self.n_nodes}, edges={self.n_edges}, "
            f"{conn}, degree={self.degree_min:.1f}/{self.degree_mean:.1f}/{self.degree_max:.1f})"
        )


# ── Perturbation-theory helpers (fully vectorized) ────────────────────────────

def eigenvalue_perturbation(V: np.ndarray, EV: np.ndarray) -> np.ndarray:
    """First-order eigenvalue shifts: δλᵢ ≈ vᵢᵀ E vᵢ."""
    return np.sum(V * EV, axis=0)


def eigenvector_perturbation(V: np.ndarray, EV: np.ndarray,
                              eigvals: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """First-order eigenvector corrections (Theorem 2 summation)."""
    Vt_EV = V.T @ EV
    diff  = eigvals[:, None] - eigvals[None, :]
    M     = np.divide(1.0, diff, out=np.zeros_like(diff), where=np.abs(diff) > tol)
    np.fill_diagonal(M, 0.0)
    return V @ (Vt_EV * M)


def projector_embedding_lowrank(V: np.ndarray, delta_V: np.ndarray,
                                  eigvals: np.ndarray, EV: np.ndarray,
                                  tol: float = 1e-10) -> np.ndarray:
    """Low-rank projector correction to the embedding."""
    V_pert = V + delta_V
    term1  = V @ (V.T @ V_pert)
    Vt_EV  = V.T @ EV
    diff   = eigvals[:, None] - eigvals[None, :]
    M      = np.divide(1.0, diff, out=np.zeros_like(diff), where=np.abs(diff) > tol)
    np.fill_diagonal(M, 0.0)
    return term1 - V @ (Vt_EV @ M) - EV @ M


# ── Graph diagnostics ─────────────────────────────────────────────────────────

def diagnose_knn_graph(A: sp.spmatrix) -> GraphDiagnostics:
    """Return connectivity diagnostics for a k-NN adjacency matrix."""
    from scipy.sparse.csgraph import connected_components
    n_components, labels = connected_components(A, directed=False)
    degrees = np.array(A.sum(axis=1)).ravel()
    return GraphDiagnostics(
        n_nodes        = A.shape[0],
        n_edges        = int(A.nnz // 2),
        n_components   = int(n_components),
        is_connected   = n_components == 1,
        degree_min     = float(degrees.min()),
        degree_mean    = float(degrees.mean()),
        degree_max     = float(degrees.max()),
        component_sizes= np.bincount(labels).tolist(),
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

class DPLaplacianEigenmaps:
    """
    Differentially private Laplacian Eigenmaps.

    Expects X to be **already scaled** (no internal StandardScaler).
    """

    def __init__(
        self,
        n_neighbors     : int                       = 10,
        n_components    : int                       = 2,
        noise_mechanism : BaseNoiseMechanism | None = None,
        normalized      : bool                      = True,
        verbose         : bool                      = False,
    ):
        self.n_neighbors     = n_neighbors
        self.n_components    = n_components
        self.noise_mechanism = noise_mechanism
        self.normalized      = normalized
        self.verbose         = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DP-LE] {msg}")

    @staticmethod
    def _fiedler_gap(eigvals: np.ndarray) -> float:
        s = np.sort(eigvals)
        return float(abs(s[1] - s[0]))

    def fit_transform(self, X: np.ndarray) -> EmbeddingResult:
        t0 = time.perf_counter()

        # 1. k-NN graph
        A = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                              mode="distance", include_self=False, n_jobs=-1)
        A = 0.5 * (A + A.T)

        # 2. Laplacian
        L = laplacian(A, normed=self.normalized)

        # 3. Clean eigenvectors
        eigvals_raw, eigvecs_raw = eigsh(L, k=self.n_components + 1, sigma=0)
        idx              = np.argsort(eigvals_raw)
        eigvals_full     = eigvals_raw[idx]
        eigvals          = eigvals_full[1:]
        eigvecs          = eigvecs_raw[:, idx][:, 1:]
        fiedler_gap_clean = self._fiedler_gap(eigvals_full)

        # 4. Noise
        if self.noise_mechanism is None:
            result = EmbeddingResult(
                embedding_clean    = eigvecs,
                embedding_noisy    = eigvecs.copy(),
                embedding_projector= eigvecs.copy(),
                eigenvalues        = eigvals,
                eigenvalues_noisy  = eigvals.copy(),
                delta_vs           = np.zeros_like(eigvecs),
                fiedler_gap_clean  = fiedler_gap_clean,
                fiedler_gap_noisy  = fiedler_gap_clean,
                L                  = L,
                noise_metadata     = None,
            )
        else:
            E, meta = self.noise_mechanism.generate(L, eigvals, eigvecs)

            # Perturbation-theory projections
            EV                  = E @ eigvecs
            delta_V             = eigenvector_perturbation(eigvecs, EV, eigvals)
            embedding_projector = projector_embedding_lowrank(eigvecs, delta_V, eigvals, EV)

            # Sparsify E onto L's sparsity pattern, then recompute eigvecs
            if not sp.issparse(E):
                E = sp.csr_matrix(E)
            L_csr    = L.tocsr()
            mask     = L_csr != 0
            E_sparse = E.tocsr().multiply(mask)
            E_sparse = (E_sparse + E_sparse.T).multiply(0.5)
            E_sparse.setdiag(0)
            E_sparse.eliminate_zeros()
            diag_fix = np.array(-E_sparse.sum(axis=1)).ravel()
            E_sparse = E_sparse + sp.diags(diag_fix)
            L_noisy  = (L_csr + E_sparse).tocsr()

            ev_n, ev_vecs      = eigsh(L_noisy, k=self.n_components + 1, sigma=0)
            idx_n              = np.argsort(ev_n)
            eigvals_noisy_full = ev_n[idx_n]
            eigvals_noisy      = eigvals_noisy_full[1:]
            embedding_noisy    = ev_vecs[:, idx_n][:, 1:]
            fiedler_gap_noisy  = self._fiedler_gap(eigvals_noisy_full)

            # Embedding-space perturbation (no-op for L-space mechanisms)
            embedding_noisy, emb_meta = self.noise_mechanism.perturb_embedding(
                embedding_noisy)
            if emb_meta is not None:
                meta = emb_meta   # EmbeddingPerturbation: replace with full metadata

            # Enrich with Fiedler fields
            meta.fiedler_gap_clean = fiedler_gap_clean
            meta.fiedler_gap_noisy = fiedler_gap_noisy
            meta.fiedler_gap_delta = fiedler_gap_noisy - fiedler_gap_clean
            meta.fiedler_gap_ratio = (
                fiedler_gap_noisy / fiedler_gap_clean if fiedler_gap_clean > 0 else float("inf")
            )

            result = EmbeddingResult(
                embedding_clean    = eigvecs,
                embedding_noisy    = embedding_noisy,
                embedding_projector= embedding_projector,
                eigenvalues        = eigvals,
                eigenvalues_noisy  = eigvals_noisy,
                delta_vs           = delta_V,
                fiedler_gap_clean  = fiedler_gap_clean,
                fiedler_gap_noisy  = fiedler_gap_noisy,
                L                  = L,
                noise_metadata     = meta,
            )

        self._log(
            f"Fiedler: clean={fiedler_gap_clean:.4e} "
            f"noisy={result.fiedler_gap_noisy:.4e} "
            f"Δ={result.fiedler_gap_noisy - fiedler_gap_clean:+.4e} "
            f"| {time.perf_counter() - t0:.2f}s"
        )
        return result
