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
    M = np.divide(1.0, diff, where=np.abs(diff) > tol)
    np.fill_diagonal(M, 0.0)

    return V @ (Vt_EV * M)


def projector_embedding_lowrank(V, delta_V, eigvals, EV, tol=1e-10):

    V_pert = V + delta_V

    term1 = V @ (V.T @ V_pert)

    Vt_EV = V.T @ EV

    diff = eigvals[:, None] - eigvals[None, :]
    M = np.divide(1.0, diff, where=np.abs(diff) > tol)
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
