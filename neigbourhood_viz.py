import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def _reduce_2d(X, method="umap", random_state=42):
    """Project X to 2D using UMAP (preferred) or t-SNE."""
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15)
        return reducer.fit_transform(X), "UMAP"
    reducer = TSNE(n_components=2, random_state=random_state, perplexity=30, init="pca")
    return reducer.fit_transform(X), "t-SNE"


def _noisy_knn_graph_from_L(L_noisy):
    """
    Reconstruct an adjacency-like matrix from a (possibly perturbed) Laplacian.
    For unnormalized L = D - A, we have A = diag(L) * I - L on off-diagonals,
    but since we use normalized Laplacians, just take the off-diagonal magnitude
    as edge weight: stronger negative off-diagonals = stronger edges.
    """
    L_dense = L_noisy.toarray() if sp.issparse(L_noisy) else np.asarray(L_noisy)
    A = -L_dense.copy()
    np.fill_diagonal(A, 0)
    A[A < 0] = 0  # discard noise-induced sign flips for edge interpretation
    return A


def _draw_graph_on_coords(ax, coords, A, y, title, edge_alpha=0.15,
                          edge_color="gray", max_edges=2000):
    """Draw a graph: nodes at coords colored by y, edges from A."""
    # Edges: take upper triangle nonzeros
    A_coo = sp.triu(sp.csr_matrix(A), k=1).tocoo()
    n_edges = A_coo.nnz

    # Subsample edges if too many (visualization gets unreadable beyond ~2k)
    if n_edges > max_edges:
        idx = np.random.default_rng(0).choice(n_edges, size=max_edges, replace=False)
        rows, cols, vals = A_coo.row[idx], A_coo.col[idx], A_coo.data[idx]
    else:
        rows, cols, vals = A_coo.row, A_coo.col, A_coo.data

    # Normalize edge weights for alpha scaling
    if len(vals) > 0:
        w_norm = vals / (vals.max() + 1e-12)
        for i, j, w in zip(rows, cols, w_norm):
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=edge_color, alpha=edge_alpha * w, linewidth=0.5, zorder=1,
            )

    # Nodes
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=y, cmap="tab10",
        s=15, alpha=0.9, edgecolors="white", linewidth=0.3, zorder=2,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter


def visualize_graph_perturbation(
    X, y, model_results, target_names=None,
    method="umap", coords=None, figsize=(15, 5),
):
    """
    Side-by-side visualization of the k-NN graph before vs after noise injection.

    Parameters
    ----------
    X : original feature matrix (will be standardized)
    y : labels for coloring
    model_results : dict returned by DPLaplacianEigenmaps.fit_transform
    coords : optional precomputed 2D coords (reuse across epsilons for fair comparison)
    """
    X_scaled = StandardScaler().fit_transform(X)

    # 2D layout — compute once and reuse so visualizations across epsilons align
    if coords is None:
        coords, method_name = _reduce_2d(X_scaled, method=method)
    else:
        method_name = method.upper()

    # Clean k-NN adjacency (rebuild from X to get edge weights, not Laplacian)
    A_clean = kneighbors_graph(
        X_scaled, n_neighbors=10, mode="connectivity", include_self=False
    )
    A_clean = 0.5 * (A_clean + A_clean.T)

    # Noisy "adjacency" reconstructed from the perturbed Laplacian
    L_clean = model_results["L"]
    # The model doesn't return L_noisy directly — reconstruct intent from the embedding shift
    # Simplest: use the difference in eigenstructure to show perturbation
    A_noisy_proxy = _noisy_knn_graph_from_L(L_clean)  # placeholder if L_noisy not in results

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: clean graph
    _draw_graph_on_coords(
        axes[0], coords, A_clean.toarray(), y,
        title=f"Clean k-NN graph\n({method_name} layout)",
        edge_color="steelblue", edge_alpha=0.25,
    )

    # Panel 2: noisy graph
    eps = model_results.get("noise_metadata", {}).get("scale", "?")
    _draw_graph_on_coords(
        axes[1], coords, A_noisy_proxy, y,
        title=f"Perturbed Laplacian structure\n(noise scale={eps:.3e})"
              if isinstance(eps, float) else "Perturbed Laplacian structure",
        edge_color="crimson", edge_alpha=0.25,
    )

    # Panel 3: edge difference (added/removed)
    A_clean_dense = A_clean.toarray()
    diff = A_noisy_proxy - A_clean_dense
    added = np.where(diff > 0.01, diff, 0)
    removed = np.where(diff < -0.01, -diff, 0)

    # Draw removed edges in blue, added in red
    A_removed_sp = sp.csr_matrix(removed)
    A_added_sp = sp.csr_matrix(added)

    _draw_graph_on_coords(
        axes[2], coords, A_removed_sp.toarray(), y,
        title="Edge changes\n(blue=removed, red=added)",
        edge_color="steelblue", edge_alpha=0.4,
    )
    # Overlay added edges
    A_added_coo = sp.triu(A_added_sp, k=1).tocoo()
    for i, j in zip(A_added_coo.row, A_added_coo.col):
        axes[2].plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color="crimson", alpha=0.4, linewidth=0.5, zorder=1,
        )

    plt.tight_layout()
    return fig, coords


def visualize_embedding_shift(
    model_results, y, target_names=None,
    method="umap", figsize=(12, 5),
):
    """
    Side-by-side: clean embedding vs noisy embedding (the actual model output).
    If n_components > 2, project both to 2D using the same reducer for comparability.
    """
    emb_clean = model_results["embedding_clean"]
    emb_noisy = model_results["embedding_noisy"]

    # If already 2D, plot directly; otherwise reduce
    if emb_clean.shape[1] == 2:
        coords_c, coords_n = emb_clean, emb_noisy
        method_name = "raw"
    else:
        # Fit one reducer on clean, transform both — but UMAP/t-SNE don't support
        # transform on new data well, so fit separately and label honestly
        coords_c, method_name = _reduce_2d(emb_clean, method=method, random_state=42)
        coords_n, _ = _reduce_2d(emb_noisy, method=method, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, coords, title in [
        (axes[0], coords_c, f"Clean embedding ({method_name})"),
        (axes[1], coords_n, f"Noisy embedding ({method_name})"),
    ]:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab10",
                        s=20, alpha=0.85, edgecolors="white", linewidth=0.3)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    if target_names is not None:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=plt.cm.tab10(i / max(len(target_names) - 1, 1)),
                       markersize=8, label=name)
            for i, name in enumerate(target_names)
        ]
        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.12, 0.5))

    plt.tight_layout()
    return fig
