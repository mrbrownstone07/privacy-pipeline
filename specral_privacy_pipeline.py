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
# 🔹 Embedding Perturbation
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
        snr_target  : float = 2.0,
        min_weight  : float = 0.05,
        random_state: int   = 42,
    ):
        self.epsilon     = epsilon
        self.delta       = delta
        self.y_train     = y_train
        self.n_train     = n_train
        self.snr_target  = snr_target
        self.min_weight  = min_weight
        self.rng         = np.random.default_rng(random_state)

    def _mi_column_weights(self, V: np.ndarray) -> np.ndarray:
        V_tr = V[:self.n_train]
        mi   = mutual_info_classif(V_tr, self.y_train,
                   discrete_features=False, random_state=42)
        mi   = np.maximum(mi, self.min_weight * (mi.max() + 1e-10))
        return mi / (mi.sum() + 1e-10)

    def perturb(self, V: np.ndarray) -> tuple:
        n, k = V.shape

        col_std  = V.std(axis=0)
        weights  = self._mi_column_weights(V)

        scales        = (weights * col_std * k) / (self.epsilon * self.snr_target)
        laplace_noise = np.zeros((n, k))
        for i in range(k):
            laplace_noise[:, i] = self.rng.laplace(0.0, scales[i], size=n)

        sigma_floor    = (self.min_weight * col_std.mean()) / (self.epsilon * self.snr_target)
        gaussian_noise = self.rng.normal(0.0, sigma_floor, size=(n, k))

        V_pert     = V + laplace_noise + gaussian_noise
        actual_snr = col_std / (scales * np.sqrt(2) + sigma_floor)

        meta = {
            "epsilon"        : self.epsilon,
            "snr_target"     : self.snr_target,
            "col_std"        : col_std.tolist(),
            "laplace_scales" : scales.tolist(),
            "gaussian_sigma" : round(float(sigma_floor), 8),
            "actual_snr"     : actual_snr.tolist(),
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
        self.rng     = np.random.default_rng(random_state)

    def _spectral_gap(self, eigvals, tol=1e-10):
        gaps  = np.diff(np.sort(eigvals))
        valid = gaps[gaps > tol]
        if len(valid) == 0:
            raise ValueError("Degenerate spectrum")
        return np.min(valid)

    def generate(self, L, eigvals, eigvecs):
        n, k = eigvecs.shape

        gap         = self._spectral_gap(eigvals)

        # Signal scale: eigenvector entries are O(1/√n)
        # Target noise_to_signal ratio = 1/snr_target
        # scale = (1/√n) / snr_target / ε
        # signal_scale = 1.0 / np.sqrt(n)
        # snr_target   = 3.0          # noise = 1/3 of signal → meaningful but not destructive
        # scale        = signal_scale / (snr_target * self.epsilon)

        ## Changed to anchor noise scale to actual signal magnitude, not a fixed formula
        signal_scale = 1.0 / np.sqrt(n)
        snr_target = 0.625   # noise ≈ signal at ε=1, stronger privacy across the sweep
        ## end of change

        scale        = signal_scale / (snr_target * self.epsilon)
        upper = self.rng.laplace(0, scale, size=(n, n))
        upper = np.triu(upper, 1)
        E     = upper + upper.T
        np.fill_diagonal(E, -E.sum(axis=1))

        noise_to_signal = scale * np.sqrt(n)   # = 1/(snr_target * ε)

        return sp.csr_matrix(E), {
            "type"            : "spectral_gap",
            "scale"           : scale,
            "gap"             : gap,
            "snr_target"      : snr_target,
            "noise_to_signal" : noise_to_signal,
            "noise_to_gap"    : scale / gap,
        }


# ============================================================
# 🔹 Resolvent Guided Perturbation
# ============================================================

class ResolventGuidedPerturbation(BaseNoiseMechanism):
    """
    Perturb a graph Laplacian using first-order perturbation theory.

    Grounded in Greenbaum, Li & Overton (2019), Theorems 1 & 2:
        δλᵢ  ≈  uᵢᵀ E uᵢ                              (Theorem 1)
        δuᵢ  ≈  -Σⱼ≠ᵢ [uⱼᵀ E uᵢ / (λᵢ - λⱼ)] uⱼ     (Theorem 2)

    Parameters
    ----------
    noise_scale       : directly controls σ of the Laplace noise injected
                        into L. This is the primary sweep parameter (ε).
    distortion_budget : α — warns when predicted ‖δuᵢ‖ exceeds this value,
                        signalling that the first-order regime has been left.
                        Does NOT cap noise — it is a diagnostic threshold only.
    random_state      : RNG seed
    """

    def __init__(
        self,
        noise_scale      : float = 1.0,
        distortion_budget: float = 0.5,
        random_state     : int   = None,
    ):
        self.noise_scale       = noise_scale        # PRIMARY sweep parameter
        self.distortion_budget = distortion_budget  # diagnostic threshold only
        self.rng               = np.random.default_rng(random_state)

    # ── Private helpers ────────────────────────────────────────────────────

    def _resolvent_norms(self, eigvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-eigenvector resolvent norms: ‖Sᵢ‖ = 1 / min_{j≠i} |λᵢ - λⱼ|
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
        Symmetric, zero-row-sum noise matrix E.
        Symmetry  → L + E remains symmetric.
        Zero row-sum → all-ones vector stays in the null space.
        """
        upper = self.rng.laplace(0, sigma, size=(n, n))
        upper = np.triu(upper, 1)
        E     = upper + upper.T
        np.fill_diagonal(E, -E.sum(axis=1))
        return E

    def _verify_first_order(
        self,
        E           : np.ndarray,
        eigvals     : np.ndarray,
        eigvecs     : np.ndarray,
        min_gaps    : np.ndarray,
        L_perturbed : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Theorem 1: predicted shift  δλᵢ ≈ uᵢᵀ E uᵢ
        Theorem 2: predicted bound  ‖δuᵢ‖ ≤ ‖Sᵢ‖ · ‖E‖_F

        first_order_valid = False signals the perturbation has left the
        linear regime; downstream eigenvector changes are then non-linear
        and the Taylor approximation cannot be trusted.
        """
        k = eigvals.shape[0]

        predicted_shifts  = np.array([eigvecs[:, i] @ E @ eigvecs[:, i] for i in range(k)])
        new_eigvals       = np.linalg.eigvalsh(L_perturbed)[:k]
        actual_shifts     = new_eigvals - eigvals
        shift_errors      = np.abs(actual_shifts - predicted_shifts)
        first_order_valid = bool(np.all(shift_errors < 0.1 * min_gaps))

        return predicted_shifts, actual_shifts, shift_errors, first_order_valid

    # ── Public interface ───────────────────────────────────────────────────

    def generate(
        self,
        L      : sp.spmatrix,
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
    ) -> tuple[sp.spmatrix, dict]:
        """
        Returns E (noise matrix only, not L + E).
        The pipeline is responsible for forming L + E and recomputing eigenvectors.
        """
        n, k = eigvecs.shape

        # Step 1: Resolvent norms (Theorem 2 denominators)
        min_gaps, resolvent_norms = self._resolvent_norms(eigvals)

        # Step 2: Noise scale — driven directly by noise_scale, not the budget
        # σ = noise_scale / √(n²-n)  keeps ‖E‖_F ≈ noise_scale regardless of n
        sigma = self.noise_scale

        # Step 3: Build noise matrix
        E = self._build_noise_matrix(n, sigma)

        # Step 4: First-order verification (diagnostic only)
        L_dense           = L.toarray() if sp.issparse(L) else L
        L_perturbed_dense = L_dense + E

        predicted_shifts, actual_shifts, shift_errors, first_order_valid = \
            self._verify_first_order(E, eigvals, eigvecs, min_gaps, L_perturbed_dense)

        E_norm_actual        = np.linalg.norm(E, "fro")
        predicted_distortion = resolvent_norms * E_norm_actual
        budget_exceeded      = bool(np.any(predicted_distortion > self.distortion_budget))

        if budget_exceeded:
            import warnings
            warnings.warn(
                f"noise_scale={self.noise_scale} produces max predicted distortion "
                f"{predicted_distortion.max():.4f} > budget {self.distortion_budget}. "
                f"First-order approximation may not hold."
            )

        meta = {
            "type"                 : "resolvent_guided",
            "noise_scale"          : self.noise_scale,
            "distortion_budget"    : self.distortion_budget,
            "sigma"                : sigma,
            "E_frobenius_norm"     : round(E_norm_actual, 6),
            "min_pairwise_gaps"    : min_gaps.tolist(),
            "resolvent_norms"      : resolvent_norms.tolist(),
            "predicted_eigenshifts": predicted_shifts.tolist(),
            "actual_eigenshifts"   : actual_shifts.tolist(),
            "shift_errors"         : shift_errors.tolist(),
            "predicted_distortions": predicted_distortion.tolist(),
            "first_order_valid"    : first_order_valid,
            "budget_exceeded"      : budget_exceeded,
        }

        return sp.csr_matrix(E), meta


# ============================================================
# 🔹 PPSP Noise
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
        self.epsilon    = epsilon
        self.y_train    = y_train
        self.n_train    = n_train
        self.delta_frob = delta_frob
        self.min_weight = min_weight
        self.rng        = np.random.default_rng(random_state)

    def _compute_weights(self, V):
        V_train = V[:self.n_train]
        mi      = mutual_info_classif(
            V_train, self.y_train, discrete_features=False, random_state=42
        )
        mi = np.maximum(mi, self.min_weight * mi.max())
        return mi / (mi.sum() + 1e-10)

    def _build_noise(self, V, weights, target_frob):
        n, k = V.shape
        E    = np.zeros((n, n))

        for i in range(k):
            vi   = V[:, i]
            u    = self.rng.standard_normal(n)
            u   -= (u @ vi) * vi
            norm = np.linalg.norm(u)
            if norm < 1e-10:
                continue
            u   /= norm
            amp  = self.rng.laplace(0.0, weights[i])
            E   += amp * (np.outer(u, vi) + np.outer(vi, u))

        frob = np.linalg.norm(E, 'fro')
        if frob > 1e-10:
            E *= target_frob / frob

        np.fill_diagonal(E, -E.sum(axis=1))
        return E

    def generate(self, L, eigvals, eigvecs):
        n, k    = eigvecs.shape
        weights = self._compute_weights(eigvecs)
        V_frob  = np.linalg.norm(eigvecs, 'fro')
        target_frob = (self.delta_frob / np.sqrt(self.epsilon)) * V_frob
        E = self._build_noise(eigvecs, weights, target_frob)
        return sp.csr_matrix(E), {
            "type": "ppsp", "weights": weights.tolist(), "target_frob": target_frob
        }


# ============================================================
# 🔹 Perturbation Theory (Fully Vectorized)
# ============================================================

def eigenvalue_perturbation(V, EV):
    return np.sum(V * EV, axis=0)


def eigenvector_perturbation(V, EV, eigvals, tol=1e-10):
    Vt_EV = V.T @ EV
    diff  = eigvals[:, None] - eigvals[None, :]
    M     = np.divide(1.0, diff, out=np.zeros_like(diff), where=np.abs(diff) > tol)
    np.fill_diagonal(M, 0.0)
    return V @ (Vt_EV * M)


def projector_embedding_lowrank(V, delta_V, eigvals, EV, tol=1e-10):
    V_pert = V + delta_V
    term1  = V @ (V.T @ V_pert)
    Vt_EV  = V.T @ EV
    diff   = eigvals[:, None] - eigvals[None, :]
    M      = np.divide(1.0, diff, out=np.zeros_like(diff), where=np.abs(diff) > tol)
    np.fill_diagonal(M, 0.0)
    term2  = V @ (Vt_EV @ M)
    term3  = EV @ M
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
        self.n_neighbors     = n_neighbors
        self.n_components    = n_components
        self.noise_mechanism = noise_mechanism
        self.normalized      = normalized
        self.verbose         = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[DP-LE] {msg}")

    def _fiedler_gap(self, eigvals):
        """
        Fiedler gap = λ₂ − λ₁ on sorted eigenvalues.
        For a connected graph Laplacian, λ₁ = 0 and λ₂ is the algebraic
        connectivity. Should be ≥ 0; abs() guards against floating-point noise.
        """
        sorted_eigvals = np.sort(eigvals)
        if len(sorted_eigvals) < 2:
            raise ValueError("Need at least 2 eigenvalues for Fiedler gap")
        return abs(sorted_eigvals[1] - sorted_eigvals[0])

    def fit_transform(self, X):
        t0 = time.perf_counter()

        # 1. Normalize
        X = StandardScaler().fit_transform(X)

        # 2. Graph
        A = kneighbors_graph(
            X, n_neighbors=self.n_neighbors,
            mode="distance", include_self=False, n_jobs=-1
        )
        A = 0.5 * (A + A.T)

        # 3. Laplacian
        L = laplacian(A, normed=self.normalized)

        # 4. Clean eigenvectors — L is sparse so eigsh(sigma=0) is safe
        eigvals, eigvecs = eigsh(L, k=self.n_components + 1, sigma=0)
        idx_clean    = np.argsort(eigvals)
        eigvals_full = eigvals[idx_clean]                # keep λ₁ for Fiedler gap
        eigvals      = eigvals_full[1:]                  # drop the trivial 0
        eigvecs      = eigvecs[:, idx_clean][:, 1:]

        # ── Fiedler gap on the clean Laplacian ─────────────────────────────
        fiedler_gap_clean = self._fiedler_gap(eigvals_full)

        # 5. Noise + embeddings
        if self.noise_mechanism is None:
            embedding_noisy     = eigvecs.copy()
            embedding_projector = eigvecs.copy()
            delta_l             = np.zeros_like(eigvals)
            delta_V             = np.zeros_like(eigvecs)
            eigvals_noisy       = eigvals.copy()
            eigvals_noisy_full  = eigvals_full.copy()
            meta                = None
            fiedler_gap_noisy   = fiedler_gap_clean
        else:
            E, meta = self.noise_mechanism.generate(L, eigvals, eigvecs)

            EV      = E @ eigvecs
            delta_l = eigenvalue_perturbation(eigvecs, EV)
            delta_V = eigenvector_perturbation(eigvecs, EV, eigvals)

            embedding_projector = projector_embedding_lowrank(
                eigvecs, delta_V, eigvals, EV
            )

            # ── Sparsify E onto L's pattern (existing logic) ───────────────
            if not sp.issparse(E):
                E = sp.csr_matrix(E)

            L_csr    = L.tocsr()
            E_csr    = E.tocsr()
            mask     = (L_csr != 0)
            E_sparse = E_csr.multiply(mask)
            E_sparse = E_sparse + E_sparse.T
            E_sparse = E_sparse.multiply(0.5)
            np.fill_diagonal(E_sparse.toarray(), 0)
            E_sparse = sp.csr_matrix(E_sparse)
            np.fill_diagonal(E_sparse.toarray(), 0)
            diag_fix = np.array(-E_sparse.sum(axis=1)).ravel()
            E_sparse = E_sparse + sp.diags(diag_fix)

            L_noisy = (L_csr + E_sparse).tocsr()

            eigvals_noisy_, eigvecs_ = eigsh(
                L_noisy, k=self.n_components + 1, sigma=0
            )
            idx_n              = np.argsort(eigvals_noisy_)
            eigvals_noisy_full = eigvals_noisy_[idx_n]            # keep λ₁
            eigvals_noisy      = eigvals_noisy_full[1:]
            embedding_noisy    = eigvecs_[:, idx_n][:, 1:]

            # ── Fiedler gap on the perturbed Laplacian ────────────────────
            fiedler_gap_noisy = self._fiedler_gap(eigvals_noisy_full)

            # Enrich noise metadata
            meta = dict(meta) if meta is not None else {}
            meta["fiedler_gap_clean"]  = fiedler_gap_clean
            meta["fiedler_gap_noisy"]  = fiedler_gap_noisy
            meta["fiedler_gap_delta"]  = fiedler_gap_noisy - fiedler_gap_clean
            meta["fiedler_gap_ratio"]  = (
                fiedler_gap_noisy / fiedler_gap_clean if fiedler_gap_clean > 0 else np.inf
            )

        self._log(
            f"Fiedler gap — clean: {fiedler_gap_clean:.4e} | "
            f"noisy: {fiedler_gap_noisy:.4e} | "
            f"Δ: {fiedler_gap_noisy - fiedler_gap_clean:+.4e} | "
            f"ratio: {fiedler_gap_noisy / fiedler_gap_clean if fiedler_gap_clean > 0 else float('inf'):.4f}"
        )
        self._log(f"Completed in {time.perf_counter() - t0:.2f}s")

        return {
            "embedding_clean"    : eigvecs,
            "embedding_noisy"    : embedding_noisy,
            "embedding_projector": embedding_projector,
            "eigenvalues"        : eigvals,
            "eigenvalues_noisy"  : eigvals_noisy,
            "delta_vs"           : delta_V,
            "noise_metadata"     : meta,
            "fiedler_gap_clean"  : fiedler_gap_clean,
            "fiedler_gap_noisy"  : fiedler_gap_noisy,
            "L"                  : L,
        }
