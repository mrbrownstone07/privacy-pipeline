import time
import numpy as np
from numpy.linalg import norm as matrix_norm

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# 🔹 Noise Mechanism Interface
# ============================================================

class BaseNoiseMechanism:
    def generate(self,  eigenvalues: np.ndarray, eigenvectors: np.ndarray):
        raise NotImplementedError


# ============================================================
# 1.  NOISE MECHANISM
# ============================================================

class SpectralGapNoise(BaseNoiseMechanism):
    """
    Generates a symmetric Laplace noise matrix E calibrated to the
    spectral gap of L.  E is a valid graph Laplacian perturbation:
    symmetric, zero row-sum.

    Scale formula (from Yetkin & Balli + DP sensitivity analysis):
        scale = (gap / (2k)) * (epsilon / sensitivity)
        sensitivity = 2 / n   (edge-level L1 sensitivity)

    Reference: Mülle et al. AAAI 2015; Sala et al. IMC 2011.
    """

    def __init__(self, epsilon: float = 1.0, random_state=None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(random_state)

    # ----------------------------------------------------------
    def _spectral_gap(self, eigenvalues: np.ndarray, tol: float = 1e-10) -> float:
        ev = np.sort(eigenvalues)
        gaps = np.diff(ev)
        valid = gaps[gaps > tol]
        if len(valid) == 0:
            raise ValueError("Degenerate spectrum — cannot compute spectral gap.")
        return float(np.min(valid))

    # ----------------------------------------------------------
    def generate(self, L, eigenvalues: np.ndarray, eigenvectors: np.ndarray):
        n = L.shape[0]
        k = eigenvectors.shape[1]

        gap         = self._spectral_gap(eigenvalues)
        sensitivity = 2.0 / n
        scale       = (gap / (2.0 * k)) * (self.epsilon / sensitivity)

        # Upper-triangle Laplace noise → symmetrize → valid Laplacian structure
        upper = self.rng.laplace(0.0, scale, size=(n, n))
        upper = np.triu(upper, 1)
        E_dense        = upper + upper.T
        np.fill_diagonal(E_dense, -E_dense.sum(axis=1))   # zero row-sum

        E_sparse = sp.csr_matrix(E_dense)
        meta     = {"scale": scale, "gap": gap, "epsilon": self.epsilon}
        return E_sparse, meta


# ============================================================
# 🔹 PPSP Laplacian Noise
# ============================================================

class PPSPLaplacianNoise:
    """
    Fixed version with three key corrections:

    Fix 1 — Noise scaled to EIGENVECTOR COORDINATE RANGE, not ‖L‖_F
             ‖L‖_F grows with n and edge density, making it a poor anchor.
             Instead we anchor to ‖V‖_F where V is the eigenvector matrix,
             because that is the space the adversary actually operates in.

    Fix 2 — MI weights INVERTED for privacy
             Original PPSP: high MI → more noise (correct for features)
             Here: high MI eigenvector component → the adversary uses it most
             → we must noise it MORE, which the original already does.
             BUT: we also need a minimum noise floor on ALL components,
             otherwise low-MI components are nearly clean and the adversary
             pivots to them.

    Fix 3 — Noise applied in EIGENVECTOR SPACE then lifted back to L-space
             Instead of noising L directly (which diffuses through all
             eigenvectors equally), we construct E as a targeted rank-k
             perturbation that maximally disrupts the top-k eigenvectors.
             This is the correct way to apply spectral privacy.
    """

    def __init__(
        self,
        epsilon     : float,
        y_train     : np.ndarray,
        n_train     : int,
        delta_frob  : float = 0.3,
        min_weight  : float = 0.05,   # Fix 2: noise floor per eigenvector
        random_state: int   = 42,
    ):
        self.epsilon      = epsilon
        self.delta_frob   = delta_frob
        self.y_train      = y_train
        self.n_train      = n_train
        self.min_weight   = min_weight
        self.rng          = np.random.default_rng(random_state)

    # ----------------------------------------------------------
    def _compute_eigvec_mi_weights(self, eigenvectors: np.ndarray) -> np.ndarray:
        """
        MI of each eigenvector column with y_train.
        Returns weights that sum to 1, with a noise floor of min_weight.

        The noise floor is critical: without it, the adversary simply
        switches to using the low-MI (low-noise) eigenvector dimensions.
        """
        V_train   = eigenvectors[:self.n_train]       # (n_train, k)
        k         = V_train.shape[1]

        mi_scores = mutual_info_classif(
            V_train, self.y_train,
            discrete_features=False,
            random_state=42,
        )

        # Apply noise floor: no eigenvector gets less than min_weight share
        mi_scores = np.maximum(mi_scores, self.min_weight * mi_scores.max())
        weights   = mi_scores / (mi_scores.sum() + 1e-10)
        return weights                                 # (k,)

    # ----------------------------------------------------------
    def _build_rank_k_noise(
        self,
        eigenvectors: np.ndarray,   # (n, k)
        ev_weights  : np.ndarray,   # (k,) — per-eigenvector noise weights
        target_frob : float,        # desired ‖E‖_F
    ) -> np.ndarray:
        """
        Fix 3: Build E as a targeted rank-2k symmetric perturbation.

        For each eigenvector v_i, we construct a rank-2 symmetric matrix:
            E_i = a_i * (u_i v_i^T + v_i u_i^T)

        where u_i is a random unit vector orthogonal to v_i, and a_i is
        scaled by the MI weight of eigenvector i.

        This concentrates noise exactly in the subspace the adversary uses,
        rather than spraying it across all of L uniformly.

        The resulting E is symmetric by construction.
        Zero row-sum is enforced by diagonal correction.
        """
        n, k   = eigenvectors.shape
        E      = np.zeros((n, n), dtype=np.float64)

        for i in range(k):
            vi = eigenvectors[:, i]                   # (n,)

            # Random direction orthogonal to v_i
            raw   = self.rng.standard_normal(n)
            raw  -= raw.dot(vi) * vi                  # project out v_i component
            norm  = np.linalg.norm(raw)
            if norm < 1e-10:
                continue
            ui    = raw / norm                        # unit vector ⊥ v_i

            # Laplace scalar amplitude for this component
            amplitude = self.rng.laplace(0.0, ev_weights[i])

            # Rank-2 symmetric update
            E    += amplitude * (np.outer(ui, vi) + np.outer(vi, ui))

        # Rescale to desired Frobenius norm
        E_frob = float(matrix_norm(E, 'fro'))
        if E_frob > 1e-10:
            E *= target_frob / E_frob

        # Zero row-sum: diagonal correction (valid Laplacian perturbation)
        np.fill_diagonal(E, -E.sum(axis=1))

        return E

    # ----------------------------------------------------------
    def generate(self, L, eigenvalues: np.ndarray, eigenvectors: np.ndarray):
        n, k   = eigenvectors.shape[0], eigenvectors.shape[1]

        # Fix 1: anchor noise budget to eigenvector space, not ‖L‖_F
        V_frob       = float(matrix_norm(eigenvectors, 'fro'))
        effective_δ  = self.delta_frob / np.sqrt(self.epsilon + 1e-10)
        target_frob  = effective_δ * V_frob   # desired ‖E‖_F in V-space units

        # MI weights over eigenvectors (with noise floor)
        ev_weights   = self._compute_eigvec_mi_weights(eigenvectors)

        # Build targeted rank-k noise matrix
        E_dense      = self._build_rank_k_noise(eigenvectors, ev_weights, target_frob)

        # Diagnostics
        L_dense      = L.toarray() if sp.issparse(L) else np.array(L)
        frob_ratio   = float(matrix_norm(E_dense, 'fro')) / (float(matrix_norm(L_dense, 'fro')) + 1e-10)

        meta = {
            "method"         : "ppsp_laplacian_v2",
            "epsilon"        : self.epsilon,
            "delta_frob"     : self.delta_frob,
            "target_frob"    : round(target_frob, 6),
            "V_frob"         : round(V_frob, 6),
            "frob_ratio"     : round(frob_ratio, 6),
            "ev_weights"     : ev_weights.tolist(),
            "top_mi_eigvecs" : np.argsort(ev_weights)[::-1].tolist(),
            "was_rescaled"   : False,
        }

        return sp.csr_matrix(E_dense), meta


# ============================================================
# 2.  PERTURBATION THEORY HELPERS  (Greenbaum et al. Thm 1 & 2)
# ============================================================

def eigenvalue_perturbation(V: np.ndarray, E) -> np.ndarray:
    """
    Theorem 1:  Δλ_i = v_i^T E v_i
    V : (n, k) dense eigenvectors
    E : (n, n) sparse noise matrix
    Returns delta_lambdas : (k,)
    """
    k = V.shape[1]
    delta_lambdas = np.zeros(k)
    for i in range(k):
        vi = V[:, i]
        delta_lambdas[i] = float(vi @ (E @ vi))
    return delta_lambdas


def eigenvector_perturbation(V: np.ndarray, E, eigvals: np.ndarray) -> np.ndarray:
    """
    Theorem 2:  Δv_i = Σ_{j≠i} [ (v_j^T E v_i) / (λ_i - λ_j) ] v_j

    Numerically safe: gaps below tol are skipped (degenerate pair guard).

    V      : (n, k) dense
    E      : (n, n) sparse
    eigvals: (k,)
    Returns delta_vs : (n, k)
    """
    n, k   = V.shape
    EV     = E @ V          # (n, k)  — precomputed for efficiency
    delta_vs = np.zeros((n, k))
    tol    = 1e-10

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            gap = eigvals[i] - eigvals[j]
            if abs(gap) < tol:
                continue                    # skip degenerate pairs
            coef = float(V[:, j] @ EV[:, i]) / gap
            delta_vs[:, i] += coef * V[:, j]

    return delta_vs


def _compute_reduced_resolvent(
    eigvecs : np.ndarray,   # (n, k)
    eigvals : np.ndarray,   # (k,)
    tol     : float = 1e-10,
) -> np.ndarray:
    """
    Reduced resolvent S approximated from the k kept eigenpairs only.

    S = Σᵢ Σⱼ≠ᵢ  v_j v_j^T / (λᵢ - λⱼ)

    Shape: (n, n)
    """
    n, k = eigvecs.shape
    S    = np.zeros((n, n), dtype=np.float64)

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            gap = eigvals[i] - eigvals[j]
            if abs(gap) < tol:
                continue
            S += np.outer(eigvecs[:, j], eigvecs[:, j]) / gap

    return S


def eigenprojector_perturbation_embedding(
    V       : np.ndarray,   # (n, k)  clean eigenvectors
    delta_V : np.ndarray,   # (n, k)  Theorem 2 shifts
    eigvals : np.ndarray,   # (k,)
    E       ,               # (n, n) sparse noise matrix
    X_scaled: np.ndarray,   # (n, d) original scaled features
    L       ,               # (n, n) sparse Laplacian (unused here, kept for API)
) -> np.ndarray:
    """
    Theorem 3 — perturbed eigenprojector embedding.

    Π₀     = V V^T                           shape (n, n)  — sample-space projector
    Π'(τ₀) = -Π₀ E S - S E Π₀               shape (n, n)  — first-order correction
    Π_new  = Π₀ + Π'(τ₀)                    shape (n, n)

    The embedding coordinates for each sample are obtained by:

        coords = Π_new @ V_pert              shape (n, k)

    Why this is correct
    -------------------
    Π_new operates in sample space (n × n). V_pert = (V + ΔV) lives in
    sample space too — each column is an n-dim vector over samples.
    Π_new @ V_pert projects the perturbed basis through the perturbed
    projector, giving the k-dim coordinate of each sample in the
    jointly-perturbed subspace.

    This differs from Noisy Embedding (V + ΔV) because Π_new contains
    the resolvent S which mixes ALL eigenpair interactions — not just
    the direct eigenvector shift from Theorem 2.

    Why the old code failed
    -----------------------
    Pi_new @ X_scaled tried (n,n) @ (n,d) correctly dimension-wise,
    but then (V_pert.T @ Pi_new @ X_scaled.T) tried
        (k,n) @ (n,n) @ (d,n)  →  inner dims (n,d) mismatched.
    X_scaled lives in feature space (n,d), not sample space — mixing
    the two spaces is the root conceptual error.
    """
    n, k    = V.shape
    E_dense = E.toarray() if sp.issparse(E) else np.asarray(E, dtype=np.float64)

    # ── Step 1: Clean eigenprojector Π₀ = V V^T ──────────────────────
    Pi0     = V @ V.T                               # (n, n)

    # ── Step 2: Reduced resolvent S ───────────────────────────────────
    S       = _compute_reduced_resolvent(V, eigvals) # (n, n)

    # ── Step 3: Π'(τ₀) = -Π₀ E S - S E Π₀  (Greenbaum eq. 10) ───────
    ES      = E_dense @ S                           # (n, n)
    SE      = S @ E_dense                           # (n, n)
    dPi     = -(Pi0 @ ES) - (SE @ Pi0)              # (n, n)

    # ── Step 4: Π_new = Π₀ + Π'(τ₀) ─────────────────────────────────
    Pi_new  = Pi0 + dPi                             # (n, n)

    # ── Step 5: Embed — project perturbed basis through Π_new ─────────
    # Both Pi_new and V_pert live in sample space (n, n) and (n, k).
    # Result: (n, n) @ (n, k) → (n, k)  ✓ correct dimensions
    V_pert  = V + delta_V                           # (n, k)
    coords  = Pi_new @ V_pert                       # (n, k)

    return coords


# ============================================================
# 3.  DP LAPLACIAN EIGENMAPS
# ============================================================

class DPLaplacianEigenmaps:
    """
    Differentially-private Laplacian Eigenmaps.

    fit_transform() returns a dict with all intermediate products and
    four embeddings so downstream code can pick the stage it needs.
    """

    def __init__(
        self,
        n_neighbors     : int   = 10,
        n_components    : int   = 2,
        normalized      : bool  = True,
        noise_mechanism : BaseNoiseMechanism | None = None,
        random_state          = None,
        verbose         : bool  = False,
    ):
        self.n_neighbors    = n_neighbors
        self.n_components   = n_components
        self.normalized     = normalized
        self.noise_mechanism = noise_mechanism
        self.random_state   = random_state
        self.verbose        = verbose

    # ------ internal helpers ----------------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(f"[DP-LE] {msg}")

    # ------ graph + Laplacian ---------------------------------
    def _build_graph(self, X: np.ndarray):
        t0 = time.perf_counter()
        X  = StandardScaler().fit_transform(X)
        A  = kneighbors_graph(
            X, n_neighbors=self.n_neighbors,
            mode="distance", include_self=False, n_jobs=-1,
        )
        A  = 0.5 * (A + A.T)
        self._log(f"Graph  shape={A.shape}  nnz={A.nnz}  ({time.perf_counter()-t0:.2f}s)")
        return A

    def _compute_affinity(self, A):
        t0    = time.perf_counter()
        sigma = np.maximum(A.max(axis=1).toarray().ravel(), 1e-10)
        rows, cols = A.nonzero()
        weights    = np.exp(-(A.data ** 2) / (sigma[rows] * sigma[cols]))
        W          = A.copy()
        W.data     = weights
        self._log(f"Affinity nnz={W.nnz}  ({time.perf_counter()-t0:.2f}s)")
        return W

    def _compute_degree(self, W):
        degrees = np.array(W.sum(axis=1)).ravel()
        degrees[degrees == 0] = 1e-10
        return sp.diags(degrees), degrees

    def _compute_laplacian(self, W, D, degrees):
        if self.normalized:
            D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
            L = D_inv_sqrt @ (D - W) @ D_inv_sqrt
            self._log("Normalized Laplacian")
        else:
            L = D - W
            self._log("Unnormalized Laplacian")
        return L

    def _eigendecompose(self, L):
        t0 = time.perf_counter()
        k  = self.n_components + 1          # +1 to skip trivial zero eigenvalue
        eigvals, eigvecs = eigsh(L, k=k, sigma=0, which="LM")

        # sort ascending — eigsh with sigma=0 is usually already sorted, but be safe
        idx     = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # drop the first (zero) eigenvalue / constant eigenvector
        eigvals = eigvals[1:]
        eigvecs = eigvecs[:, 1:]
        self._log(f"Eigenvalues={np.round(eigvals,6)}  ({time.perf_counter()-t0:.2f}s)")
        return eigvals, eigvecs

    # ------ public API ----------------------------------------
    def fit_transform(self, X: np.ndarray) -> dict:
        t_total = time.perf_counter()
        self._log("=== DPLaplacianEigenmaps.fit_transform ===")

        # Graph + Laplacian
        A            = self._build_graph(X)
        W            = self._compute_affinity(A)
        D, degrees   = self._compute_degree(W)
        L            = self._compute_laplacian(W, D, degrees)

        # Clean eigendecomposition
        eigvals, eigvecs = self._eigendecompose(L)

        # ---- noise + perturbation ----------------------------
        if self.noise_mechanism is None:
            self._log("No noise applied.")
            E            = None
            noise_meta   = None
            delta_lambdas = np.zeros(self.n_components)
            delta_vs      = np.zeros_like(eigvecs)
        else:
            t0 = time.perf_counter()
            E, noise_meta = self.noise_mechanism.generate(L, eigvals, eigvecs)

            # Theorem 1: eigenvalue shifts
            delta_lambdas = eigenvalue_perturbation(eigvecs, E)

            # Theorem 2: eigenvector shifts
            delta_vs      = eigenvector_perturbation(eigvecs, E, eigvals)

            # self._log(
            #     f"Noise applied  scale={noise_meta['scale']:.4e}"
            #     f"  gap={noise_meta['gap']:.4e}"
            #     f"  ({time.perf_counter()-t0:.2f}s)"
            # )

        # ---- four embeddings ---------------------------------
        #  (a) clean embedding   — raw eigenvectors
        embedding_clean      = eigvecs                          # (n, k)

        #  (b) noisy embedding   — eigenvectors + Δv  (Theorem 2)
        embedding_noisy      = eigvecs + delta_vs               # (n, k)

        #  (c) eigen-projector   — perturbed projector basis   (Theorem 3, FIXED)
        #      Previously this applied the projector to X_features which is WRONG.
        #      The correct embedding is simply V + ΔV as the perturbed basis.
        embedding_projector = eigenprojector_perturbation_embedding(
            V        = eigvecs,
            delta_V  = delta_vs,
            eigvals  = eigvals,
            E        = E,               # the noise matrix from noise_mechanism.generate()
            X_scaled = X,               # the input passed to fit_transform
            L        = L,
        )

        self._log(f"Total time: {time.perf_counter()-t_total:.2f}s")

        return {
            # ---- embeddings (all shape: n_samples × n_components) ----
            "embedding_clean"     : embedding_clean,
            "embedding_noisy"     : embedding_noisy,
            "embedding_projector" : embedding_projector,

            # ---- eigen products ----
            "eigenvalues"         : eigvals,
            "eigenvalues_noisy"   : eigvals + delta_lambdas,
            "eigenvectors"        : eigvecs,
            "delta_vs"            : delta_vs,
            "delta_lambdas"       : delta_lambdas,

            # ---- metadata ----
            "noise_metadata"      : noise_meta,
            "L"                   : L,
        }
