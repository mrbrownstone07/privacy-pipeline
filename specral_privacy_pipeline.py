import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import time


# ============================================================
# 🔹 Noise Interface
# ============================================================

class BaseNoiseMechanism:
    """
    All noise mechanisms must implement this interface.
    """

    def generate(self, L, eigvals, eigvecs):
        """
        This method will be called by the main pipeline to generate noise.

        Returns:
            E    : (n, n) sparse noise matrix
            meta : dict with diagnostics
        """
        raise NotImplementedError


# ============================================================
# 🔹Embedding Perturbation
# ============================================================

class EmbeddingPerturbation:
    """
    MI-weighted + Gaussian output perturbation on the clean embedding V.

    Core fix vs previous version
    ----------------------------
    Noise scale is now anchored to the ACTUAL coordinate range of each
    eigenvector column, not a fixed formula borrowed from feature space.

    For eigenvectors:  entries are O(1/√n),  std per column ≈ 1/√n
    Correct scale:     noise_std_i = (w_i * std_i * amplifier) / epsilon

    This ensures the signal-to-noise ratio is controlled, not the
    absolute noise magnitude.
    """

    def __init__(
        self,
        epsilon     : float,
        y_train     : np.ndarray,
        n_train     : int,
        delta       : float = 1e-5,
        snr_target  : float = 2.0,    # signal / noise ratio target
        min_weight  : float = 0.05,
        random_state: int   = 42,
    ):
        self.epsilon     = epsilon
        self.delta       = delta
        self.y_train     = y_train
        self.n_train     = n_train
        self.snr_target  = snr_target   # higher = more utility, less privacy
        self.min_weight  = min_weight
        self.rng         = np.random.default_rng(random_state)

    def _mi_column_weights(self, V: np.ndarray) -> np.ndarray:
        from sklearn.feature_selection import mutual_info_classif
        V_tr    = V[:self.n_train]
        mi      = mutual_info_classif(V_tr, self.y_train,
                      discrete_features=False, random_state=42)
        mi      = np.maximum(mi, self.min_weight * (mi.max() + 1e-10))
        return mi / (mi.sum() + 1e-10)

    def perturb(self, V: np.ndarray) -> tuple:
        n, k = V.shape

        # ── Per-column stats anchored to actual V scale ────────────────
        col_std  = V.std(axis=0)          # (k,)  typically O(1/√n)

        # ── MI weights ────────────────────────────────────────────────
        weights  = self._mi_column_weights(V)

        # ── Laplace scales anchored to col_std ────────────────────────
        # scale_i = (w_i * col_std_i * k) / (epsilon * snr_target)
        # snr_target controls the signal/noise tradeoff directly:
        #   snr=1 → noise_std ≈ signal_std  (strong privacy, low utility)
        #   snr=5 → noise_std ≈ 0.2×signal  (mild privacy, good utility)
        scales        = (weights * col_std * k) / (self.epsilon * self.snr_target)
        laplace_noise = np.zeros((n, k))
        for i in range(k):
            laplace_noise[:, i] = self.rng.laplace(0.0, scales[i], size=n)

        # ── Gaussian floor anchored to col_std ────────────────────────
        # sigma = (min_weight * col_std.mean()) / (epsilon * snr_target)
        # This ensures the floor never overwhelms signal
        sigma_floor   = (self.min_weight * col_std.mean()) / (self.epsilon * self.snr_target)
        gaussian_noise = self.rng.normal(0.0, sigma_floor, size=(n, k))

        V_pert = V + laplace_noise + gaussian_noise

        # Diagnostics — the key ones to watch
        actual_snr = col_std / (scales * np.sqrt(2) + sigma_floor)   # approx SNR per col

        meta = {
            "epsilon"        : self.epsilon,
            "snr_target"     : self.snr_target,
            "col_std"        : col_std.tolist(),
            "laplace_scales" : scales.tolist(),
            "gaussian_sigma" : round(float(sigma_floor), 8),
            "actual_snr"     : actual_snr.tolist(),    # watch this — should be > 1
            "mi_weights"     : weights.tolist(),
            "frob_ratio"     : round(
                float(np.linalg.norm(V_pert, 'fro')) /
                float(np.linalg.norm(V, 'fro')), 4),
        }
        return V_pert, meta

# ============================================================
# 🔹 Spectral Gap Noise (baseline DP)
# ============================================================

class SpectralGapNoise(BaseNoiseMechanism):

    def __init__(self, epsilon=1.0, random_state=None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(random_state)

    def _spectral_gap(self, eigvals, tol=1e-10):
        gaps = np.diff(np.sort(eigvals))
        valid = gaps[gaps > tol]
        if len(valid) == 0:
            raise ValueError("Degenerate spectrum")
        return np.min(valid)

    def generate(self, L, eigvals, eigvecs):
        n, k = eigvecs.shape

        gap = self._spectral_gap(eigvals)
        sensitivity = 2.0 / n
        scale = (gap / (2 * k)) * (self.epsilon / sensitivity)

        upper = self.rng.laplace(0, scale, size=(n, n))
        upper = np.triu(upper, 1)
        E = upper + upper.T
        np.fill_diagonal(E, -E.sum(axis=1))

        return sp.csr_matrix(E), {"type": "spectral_gap", "scale": scale, "gap": gap}


# ============================================================
# 🔹 Per-class metrics with low-support fla
# ============================================================
class ResolventGuidedPerturbation(BaseNoiseMechanism):
    """
    Perturb a graph Laplacian using first-order perturbation theory.

    Grounded in Greenbaum, Li & Overton (2019), Theorems 1 & 2:
        δλᵢ  ≈  uᵢᵀ E uᵢ                              (Theorem 1)
        δuᵢ  ≈  -Σⱼ≠ᵢ [uⱼᵀ E uᵢ / (λᵢ - λⱼ)] uⱼ     (Theorem 2)

    The noise scale is derived by bounding ‖E‖ via the resolvent norm:
        ‖E‖ ≤ distortion_budget · min_i [ min_{j≠i} |λᵢ - λⱼ| ]
    """

    def __init__(self, distortion_budget: float = 0.1, random_state: int = None):
        """
        Parameters
        ----------
        distortion_budget : α — max allowable ‖δuᵢ‖ per eigenvector
        random_state      : RNG seed
        """
        self.distortion_budget = distortion_budget
        self.rng               = np.random.default_rng(random_state)

    # ── Private helpers ────────────────────────────────────────────────────

    def _resolvent_norms(self, eigvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-eigenvector resolvent norms and minimum pairwise gaps.

        From Theorem 2, the reduced resolvent Sᵢ has operator norm:
            ‖Sᵢ‖ = 1 / min_{j≠i} |λᵢ - λⱼ|

        Returns
        -------
        min_gaps       : (k,) minimum gap per eigenvector
        resolvent_norms: (k,) ‖Sᵢ‖ per eigenvector
        """
        pairwise_gaps = np.abs(eigvals[:, None] - eigvals[None, :])
        np.fill_diagonal(pairwise_gaps, np.inf)
        min_gaps = pairwise_gaps.min(axis=1)

        if np.any(min_gaps < 1e-10):
            raise ValueError(
                "Degenerate spectrum: eigenvalues too close to guarantee "
                "first-order validity. Increase spectral separation before perturbing."
            )

        return min_gaps, 1.0 / min_gaps

    def _build_noise_matrix(self, n: int, sigma: float) -> np.ndarray:
        """
        Build a symmetric, zero-row-sum noise matrix E.
        Symmetry preserves the Laplacian structure of L + E.
        Zero row-sum ensures the all-ones vector remains in the null space.
        """
        upper = self.rng.laplace(0, sigma, size=(n, n))
        upper = np.triu(upper, 1)
        E     = upper + upper.T
        np.fill_diagonal(E, -E.sum(axis=1))
        return E

    def _verify_first_order(
        self,
        E:        np.ndarray,
        eigvals:  np.ndarray,
        eigvecs:  np.ndarray,
        min_gaps: np.ndarray,
        L_perturbed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Verify the first-order regime using Theorems 1 and 2.

        Theorem 1: predicted shift  δλᵢ ≈ uᵢᵀ E uᵢ
        Theorem 2: predicted bound  ‖δuᵢ‖ ≤ ‖Sᵢ‖ · ‖E‖_F

        If shift errors exceed 10% of the local gap, the perturbation
        has left the first-order regime and results are unreliable.
        """
        k = eigvals.shape[0]

        predicted_shifts = np.array([eigvecs[:, i] @ E @ eigvecs[:, i] for i in range(k)])
        new_eigvals      = np.linalg.eigvalsh(L_perturbed)[:k]
        actual_shifts    = new_eigvals - eigvals
        shift_errors     = np.abs(actual_shifts - predicted_shifts)
        first_order_valid = bool(np.all(shift_errors < 0.1 * min_gaps))

        return predicted_shifts, actual_shifts, shift_errors, first_order_valid

    # ── Public interface ───────────────────────────────────────────────────

    def generate(
        self,
        L:       sp.spmatrix,
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
    ) -> tuple[sp.spmatrix, dict]:
        """
        Generate a resolvent-guided Laplacian perturbation matrix E.

        Parameters
        ----------
        L       : symmetric graph Laplacian (n × n sparse)
        eigvals : (k,)   eigenvalues  of L
        eigvecs : (n, k) eigenvectors of L

        Returns
        -------
        E    : sp.csr_matrix — structured noise matrix (not L + E)
        meta : dict          — resolvent norms, predicted vs actual shifts,
                               first-order validity flag
        """
        n, k = eigvecs.shape

        # Step 1: Resolvent norms from Theorem 2 denominators
        min_gaps, resolvent_norms = self._resolvent_norms(eigvals)

        # Step 2: Derive max allowable ‖E‖ and noise scale σ
        # ‖δuᵢ‖ ≤ ‖Sᵢ‖ · ‖E‖ ≤ α  →  ‖E‖ ≤ α / max_i ‖Sᵢ‖
        E_norm_max = self.distortion_budget / resolvent_norms.max()
        sigma      = E_norm_max / np.sqrt(n * (n - 1))

        # Step 3: Build symmetric zero-row-sum noise matrix
        E = self._build_noise_matrix(n, sigma)

        # Step 4 & 5: First-order verification via Theorems 1 and 2
        L_dense          = L.toarray() if sp.issparse(L) else L
        L_perturbed_dense = L_dense + E

        predicted_shifts, actual_shifts, shift_errors, first_order_valid = \
            self._verify_first_order(E, eigvals, eigvecs, min_gaps, L_perturbed_dense)

        E_norm_actual = np.linalg.norm(E, "fro")

        meta = {
            "type":                  "resolvent_guided",
            "distortion_budget":     self.distortion_budget,
            "sigma":                 sigma,
            "E_frobenius_norm":      round(E_norm_actual, 6),
            "E_norm_max":            round(E_norm_max,    6),
            "min_pairwise_gaps":     min_gaps.tolist(),
            "resolvent_norms":       resolvent_norms.tolist(),
            "predicted_eigenshifts": predicted_shifts.tolist(),
            "actual_eigenshifts":    actual_shifts.tolist(),
            "shift_errors":          shift_errors.tolist(),
            "predicted_distortions": (resolvent_norms * E_norm_actual).tolist(),
            "first_order_valid":     first_order_valid,
        }

        return sp.csr_matrix(E), meta


# ============================================================
# 🔹 PPSP Noise (Optimized & Clean)
# ============================================================

class PPSPLaplacianNoise(BaseNoiseMechanism):

    def __init__(
        self,
        epsilon,
        y_train,
        n_train,
        delta_frob=0.3,
        min_weight=0.05,
        random_state=42,
    ):
        self.epsilon = epsilon
        self.y_train = y_train
        self.n_train = n_train
        self.delta_frob = delta_frob
        self.min_weight = min_weight
        self.rng = np.random.default_rng(random_state)

    # --------------------------------------------------------
    def _compute_weights(self, V):
        V_train = V[:self.n_train]

        mi = mutual_info_classif(
            V_train,
            self.y_train,
            discrete_features=False,
            random_state=42
        )

        # Noise floor (critical!)
        mi = np.maximum(mi, self.min_weight * mi.max())
        return mi / (mi.sum() + 1e-10)

    # --------------------------------------------------------
    def _build_noise(self, V, weights, target_frob):
        n, k = V.shape
        E = np.zeros((n, n))

        for i in range(k):
            vi = V[:, i]

            # Random orthogonal direction
            u = self.rng.standard_normal(n)
            u -= (u @ vi) * vi
            norm = np.linalg.norm(u)
            if norm < 1e-10:
                continue
            u /= norm

            amp = self.rng.laplace(0.0, weights[i])
            E += amp * (np.outer(u, vi) + np.outer(vi, u))

        # Normalize Frobenius norm
        frob = np.linalg.norm(E, 'fro')
        if frob > 1e-10:
            E *= target_frob / frob

        # Enforce Laplacian structure
        np.fill_diagonal(E, -E.sum(axis=1))

        return E

    # --------------------------------------------------------
    def generate(self, L, eigvals, eigvecs):
        n, k = eigvecs.shape

        weights = self._compute_weights(eigvecs)

        V_frob = np.linalg.norm(eigvecs, 'fro')
        target_frob = (self.delta_frob / np.sqrt(self.epsilon)) * V_frob

        E = self._build_noise(eigvecs, weights, target_frob)

        return sp.csr_matrix(E), {
            "type": "ppsp",
            "weights": weights.tolist(),
            "target_frob": target_frob
        }


# ============================================================
# 🔹 Perturbation Theory (Fully Vectorized)
# ============================================================

def eigenvalue_perturbation(V, EV):
    return np.sum(V * EV, axis=0)


def eigenvector_perturbation(V, EV, eigvals, tol=1e-10):
    Vt_EV = V.T @ EV

    diff = eigvals[:, None] - eigvals[None, :]
    M = np.divide(1.0, diff, out=np.zeros_like(diff), where=np.abs(diff) > tol)
    np.fill_diagonal(M, 0.0)

    return V @ (Vt_EV * M)


def projector_embedding_lowrank(V, delta_V, eigvals, EV, tol=1e-10):

    V_pert = V + delta_V

    term1 = V @ (V.T @ V_pert)

    Vt_EV = V.T @ EV

    diff = eigvals[:, None] - eigvals[None, :]
    M = np.divide(1.0, diff, out=np.zeros_like(diff), where=np.abs(diff) > tol)
    np.fill_diagonal(M, 0.0)

    term2 = V @ (Vt_EV @ M)
    term3 = EV @ M

    return term1 - term2 - term3


# ============================================================
# 🔹 Main Pipeline
# ============================================================

class DPLaplacianEigenmaps:

    def __init__(
        self,
        n_neighbors=10,
        n_components=2,
        noise_mechanism=None,
        normalized=True,
        verbose=False
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.noise_mechanism = noise_mechanism
        self.normalized = normalized
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[DP-LE] {msg}")

    def fit_transform(self, X):

        t0 = time.perf_counter()

        # 1. Normalize
        X = StandardScaler().fit_transform(X)

        # 2. Graph
        A = kneighbors_graph(
            X,
            n_neighbors=self.n_neighbors,
            mode="distance",
            include_self=False,
            n_jobs=-1
        )
        A = 0.5 * (A + A.T)

        # 3. Laplacian
        L = laplacian(A, normed=self.normalized)

        # 4. Eigen
        eigvals, eigvecs = eigsh(L, k=self.n_components + 1, sigma=0)

        idx = np.argsort(eigvals)
        eigvals = eigvals[idx][1:]
        eigvecs = eigvecs[:, idx][:, 1:]

        # 5. Noise + perturbation
        if self.noise_mechanism is None:
            EV = np.zeros_like(eigvecs)
            delta_l = np.zeros_like(eigvals)
            delta_V = np.zeros_like(eigvecs)
            meta = None
        else:
            E, meta = self.noise_mechanism.generate(L, eigvals, eigvecs)

            EV = E @ eigvecs

            delta_l = eigenvalue_perturbation(eigvecs, EV)
            delta_V = eigenvector_perturbation(eigvecs, EV, eigvals)

        # 6. Embeddings
        embedding_clean = eigvecs
        embedding_noisy = eigvecs + delta_V
        embedding_projector = projector_embedding_lowrank(
            eigvecs, delta_V, eigvals, EV
        )

        self._log(f"Completed in {time.perf_counter() - t0:.2f}s")

        return {
            "embedding_clean": embedding_clean,
            "embedding_noisy": embedding_noisy,
            "embedding_projector": embedding_projector,
            "eigenvalues": eigvals,
            "eigenvalues_noisy": eigvals + delta_l,
            "delta_vs": delta_V,
            "noise_metadata": meta,
            "L": L,
        }
