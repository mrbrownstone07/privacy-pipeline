from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.feature_selection import mutual_info_classif


# ── Metadata types ────────────────────────────────────────────────────────────

@dataclass
class NoiseMetadata:
    """Base metadata shared by all noise mechanisms.  Fiedler fields are filled
    in by DPLaplacianEigenmaps after noise is applied."""
    type: str = "base"
    fiedler_gap_clean: float | None = None
    fiedler_gap_noisy: float | None = None
    fiedler_gap_delta : float | None = None
    fiedler_gap_ratio : float | None = None


@dataclass
class SpectralGapMetadata(NoiseMetadata):
    type          : str   = "spectral_gap"
    scale         : float = 0.0
    gap           : float = 0.0
    snr_target    : float = 0.0
    noise_to_signal: float = 0.0
    noise_to_gap  : float = 0.0


@dataclass
class ResolventGuidedMetadata(NoiseMetadata):
    type                : str         = "resolvent_guided"
    noise_scale         : float       = 0.0
    distortion_budget   : float       = 0.0
    sigma               : float       = 0.0
    E_frobenius_norm    : float       = 0.0
    min_pairwise_gaps   : list[float] = field(default_factory=list)
    resolvent_norms     : list[float] = field(default_factory=list)
    predicted_eigenshifts: list[float] = field(default_factory=list)
    actual_eigenshifts  : list[float] = field(default_factory=list)
    shift_errors        : list[float] = field(default_factory=list)
    predicted_distortions: list[float] = field(default_factory=list)
    first_order_valid   : bool        = True
    budget_exceeded     : bool        = False


@dataclass
class PPSPMetadata(NoiseMetadata):
    type        : str         = "ppsp"
    weights     : list[float] = field(default_factory=list)
    target_frob : float       = 0.0


@dataclass
class EmbeddingPerturbationMetadata(NoiseMetadata):
    type          : str         = "embedding"
    epsilon       : float       = 0.0
    snr_target    : float       = 0.0
    col_std       : list[float] = field(default_factory=list)
    laplace_scales: list[float] = field(default_factory=list)
    gaussian_sigma: float       = 0.0
    actual_snr    : list[float] = field(default_factory=list)
    mi_weights    : list[float] = field(default_factory=list)
    frob_ratio    : float       = 0.0


# ── Interface ─────────────────────────────────────────────────────────────────

class BaseNoiseMechanism:
    """Interface for all noise mechanisms."""

    def generate(
        self, L: sp.spmatrix, eigvals: np.ndarray, eigvecs: np.ndarray
    ) -> tuple[sp.spmatrix, NoiseMetadata]:
        raise NotImplementedError

    def perturb_embedding(
        self, V: np.ndarray
    ) -> tuple[np.ndarray, NoiseMetadata | None]:
        """No-op default — override in embedding-space mechanisms."""
        return V, None


# ── Spectral Gap Noise (baseline DP) ─────────────────────────────────────────

class SpectralGapNoise(BaseNoiseMechanism):

    def __init__(self, epsilon: float = 1.0, snr_target: float = 0.625,
                 random_state=None):
        self.epsilon    = epsilon
        self.snr_target = snr_target
        self.rng        = np.random.default_rng(random_state)

    def _spectral_gap(self, eigvals: np.ndarray, tol: float = 1e-10) -> float:
        gaps  = np.diff(np.sort(eigvals))
        valid = gaps[gaps > tol]
        if len(valid) == 0:
            raise ValueError("Degenerate spectrum")
        return float(np.min(valid))

    def generate(
        self, L, eigvals, eigvecs
    ) -> tuple[sp.spmatrix, SpectralGapMetadata]:
        n     = eigvecs.shape[0]
        gap   = self._spectral_gap(eigvals)
        scale = (1.0 / np.sqrt(n)) / (self.snr_target * self.epsilon)

        upper = np.triu(self.rng.laplace(0, scale, size=(n, n)), 1)
        E     = upper + upper.T
        np.fill_diagonal(E, -E.sum(axis=1))

        return sp.csr_matrix(E), SpectralGapMetadata(
            scale          = scale,
            gap            = gap,
            snr_target     = self.snr_target,
            noise_to_signal= scale * np.sqrt(n),
            noise_to_gap   = scale / gap,
        )


# ── Resolvent-Guided Perturbation ─────────────────────────────────────────────

class ResolventGuidedPerturbation(BaseNoiseMechanism):
    """
    Grounded in Greenbaum, Li & Overton (2019), Theorems 1 & 2:
        δλᵢ ≈ uᵢᵀ E uᵢ            (Theorem 1)
        ‖δuᵢ‖ ≤ ‖Sᵢ‖ · ‖E‖_F      (Theorem 2)
    """

    def __init__(self, noise_scale: float = 1.0, distortion_budget: float = 0.5,
                 random_state=None):
        self.noise_scale       = noise_scale
        self.distortion_budget = distortion_budget
        self.rng               = np.random.default_rng(random_state)

    def _resolvent_norms(self, eigvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gaps = np.abs(eigvals[:, None] - eigvals[None, :])
        np.fill_diagonal(gaps, np.inf)
        min_gaps = gaps.min(axis=1)
        if np.any(min_gaps < 1e-10):
            raise ValueError(
                "Degenerate spectrum: eigenvalues too close. Increase spectral "
                "separation before perturbing."
            )
        return min_gaps, 1.0 / min_gaps

    def _build_noise_matrix(self, n: int) -> np.ndarray:
        upper = np.triu(self.rng.laplace(0, self.noise_scale, size=(n, n)), 1)
        E     = upper + upper.T
        np.fill_diagonal(E, -E.sum(axis=1))
        return E

    def _verify_first_order(
        self, E, eigvals, eigvecs, min_gaps, L_perturbed
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        EV               = E @ eigvecs
        predicted_shifts = np.sum(eigvecs * EV, axis=0)
        actual_shifts    = np.linalg.eigvalsh(L_perturbed)[:len(eigvals)] - eigvals
        shift_errors     = np.abs(actual_shifts - predicted_shifts)
        first_order_valid = bool(np.all(shift_errors < 0.1 * min_gaps))
        return predicted_shifts, actual_shifts, shift_errors, first_order_valid

    def generate(
        self, L, eigvals, eigvecs
    ) -> tuple[sp.spmatrix, ResolventGuidedMetadata]:
        n                   = eigvecs.shape[0]
        min_gaps, res_norms = self._resolvent_norms(eigvals)
        E                   = self._build_noise_matrix(n)
        L_dense             = L.toarray() if sp.issparse(L) else L

        pred_shifts, act_shifts, shift_errs, fo_valid = \
            self._verify_first_order(E, eigvals, eigvecs, min_gaps, L_dense + E)

        E_norm          = np.linalg.norm(E, "fro")
        pred_distortion = res_norms * E_norm
        budget_exceeded = bool(np.any(pred_distortion > self.distortion_budget))

        if budget_exceeded:
            warnings.warn(
                f"noise_scale={self.noise_scale}: max predicted distortion "
                f"{pred_distortion.max():.4f} > budget {self.distortion_budget}. "
                "First-order approximation may not hold."
            )

        return sp.csr_matrix(E), ResolventGuidedMetadata(
            noise_scale          = self.noise_scale,
            distortion_budget    = self.distortion_budget,
            sigma                = self.noise_scale,
            E_frobenius_norm     = round(E_norm, 6),
            min_pairwise_gaps    = min_gaps.tolist(),
            resolvent_norms      = res_norms.tolist(),
            predicted_eigenshifts= pred_shifts.tolist(),
            actual_eigenshifts   = act_shifts.tolist(),
            shift_errors         = shift_errs.tolist(),
            predicted_distortions= pred_distortion.tolist(),
            first_order_valid    = fo_valid,
            budget_exceeded      = budget_exceeded,
        )


# ── PPSP Laplacian Noise ──────────────────────────────────────────────────────

class PPSPLaplacianNoise(BaseNoiseMechanism):
    """MI-weighted noise projected orthogonally to eigenvectors."""

    def __init__(self, epsilon: float, y_train: np.ndarray, n_train: int,
                 delta_frob: float = 0.3, min_weight: float = 0.05,
                 random_state: int = 42):
        self.epsilon    = epsilon
        self.y_train    = y_train
        self.n_train    = n_train
        self.delta_frob = delta_frob
        self.min_weight = min_weight
        self.rng        = np.random.default_rng(random_state)

    def _compute_weights(self, V: np.ndarray) -> np.ndarray:
        mi = mutual_info_classif(V[:self.n_train], self.y_train,
                                 discrete_features=False, random_state=42)
        mi = np.maximum(mi, self.min_weight * mi.max())
        return mi / (mi.sum() + 1e-10)

    def _build_noise(self, V: np.ndarray, weights: np.ndarray,
                     target_frob: float) -> np.ndarray:
        n, k = V.shape
        E    = np.zeros((n, n))
        for i in range(k):
            vi = V[:, i]
            u  = self.rng.standard_normal(n)
            u -= (u @ vi) * vi
            norm = np.linalg.norm(u)
            if norm < 1e-10:
                continue
            u  /= norm
            amp = self.rng.laplace(0.0, weights[i])
            E  += amp * (np.outer(u, vi) + np.outer(vi, u))
        frob = np.linalg.norm(E, "fro")
        if frob > 1e-10:
            E *= target_frob / frob
        np.fill_diagonal(E, -E.sum(axis=1))
        return E

    def generate(
        self, L, eigvals, eigvecs
    ) -> tuple[sp.spmatrix, PPSPMetadata]:
        weights     = self._compute_weights(eigvecs)
        target_frob = (self.delta_frob / np.sqrt(self.epsilon)) * np.linalg.norm(eigvecs, "fro")
        E           = self._build_noise(eigvecs, weights, target_frob)
        return sp.csr_matrix(E), PPSPMetadata(
            weights    = weights.tolist(),
            target_frob= target_frob,
        )


# ── Embedding Perturbation ────────────────────────────────────────────────────

class EmbeddingPerturbation(BaseNoiseMechanism):
    """
    MI-weighted output perturbation applied directly to the clean embedding V.

    Noise scale is anchored to the actual column std of V (entries O(1/√n))
    so SNR is controlled rather than the absolute noise magnitude.
    """

    def __init__(self, epsilon: float, y_train: np.ndarray, n_train: int,
                 delta: float = 1e-5, snr_target: float = 2.0,
                 min_weight: float = 0.05, random_state: int = 42):
        self.epsilon    = epsilon
        self.delta      = delta
        self.y_train    = y_train
        self.n_train    = n_train
        self.snr_target = snr_target
        self.min_weight = min_weight
        self.rng        = np.random.default_rng(random_state)

    def generate(
        self, L, eigvals, eigvecs
    ) -> tuple[sp.spmatrix, EmbeddingPerturbationMetadata]:
        n = L.shape[0]
        return sp.csr_matrix((n, n)), EmbeddingPerturbationMetadata(
            epsilon   = self.epsilon,
            snr_target= self.snr_target,
        )

    def perturb_embedding(
        self, V: np.ndarray
    ) -> tuple[np.ndarray, EmbeddingPerturbationMetadata]:
        n, k   = V.shape
        mi     = mutual_info_classif(V[:self.n_train], self.y_train,
                                     discrete_features=False, random_state=42)
        mi     = np.maximum(mi, self.min_weight * (mi.max() + 1e-10))
        weights = mi / (mi.sum() + 1e-10)

        col_std = V.std(axis=0)
        scales  = (weights * col_std * k) / (self.epsilon * self.snr_target)

        laplace_noise  = self.rng.laplace(0, 1, size=(n, k)) * scales
        sigma_floor    = (self.min_weight * col_std.mean()) / (self.epsilon * self.snr_target)
        gaussian_noise = self.rng.normal(0.0, sigma_floor, size=(n, k))

        V_pert     = V + laplace_noise + gaussian_noise
        actual_snr = col_std / (scales * np.sqrt(2) + sigma_floor)

        return V_pert, EmbeddingPerturbationMetadata(
            epsilon       = self.epsilon,
            snr_target    = self.snr_target,
            col_std       = col_std.tolist(),
            laplace_scales= scales.tolist(),
            gaussian_sigma= round(float(sigma_floor), 8),
            actual_snr    = actual_snr.tolist(),
            mi_weights    = weights.tolist(),
            frob_ratio    = round(
                float(np.linalg.norm(V_pert, "fro")) /
                float(np.linalg.norm(V, "fro")), 4),
        )


# ── Factory ───────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {
    "spectral_gap"    : SpectralGapNoise,
    "resolvent_guided": ResolventGuidedPerturbation,
    "ppsp"            : PPSPLaplacianNoise,
    "embedding"       : EmbeddingPerturbation,
}


def build_noise_mechanism(
    name        : str,
    epsilon     : float,
    y           : np.ndarray | None = None,
    n_train     : int | None        = None,
    random_state: int               = 42,
    **kwargs,
) -> BaseNoiseMechanism:
    """
    Instantiate a noise mechanism by name.

    Parameters
    ----------
    name         : one of {spectral_gap, resolvent_guided, ppsp, embedding}
    epsilon      : privacy budget / noise scale
    y            : class labels — required by ppsp and embedding
    n_train      : training-set size — required by ppsp and embedding
    random_state : RNG seed
    **kwargs     : forwarded to the mechanism constructor
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown mechanism '{name}'. Choose from {list(_REGISTRY)}")

    cls = _REGISTRY[name]

    if name == "spectral_gap":
        return cls(epsilon=epsilon, random_state=random_state, **kwargs)
    if name == "resolvent_guided":
        return cls(noise_scale=epsilon, random_state=random_state, **kwargs)

    if y is None or n_train is None:
        raise ValueError(f"'{name}' requires y and n_train")
    return cls(epsilon=epsilon, y_train=y, n_train=n_train,
               random_state=random_state, **kwargs)
