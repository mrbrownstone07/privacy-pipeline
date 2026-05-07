from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from .graph import EmbeddingResult
    from .experiment import EpsilonRecord, FiedlerRecord, BaselineRecord

try:
    import umap as _umap # pyright: ignore[reportMissingImports]

    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


# ── Shared helpers ────────────────────────────────────────────────────────────


def _reduce_2d(
    X: np.ndarray, method: str = "umap", random_state: int = 42
) -> tuple[np.ndarray, str]:
    """Project X to 2D via UMAP (preferred) or t-SNE."""
    if method == "umap" and _HAS_UMAP:
        coords = _umap.UMAP(
            n_components=2, random_state=random_state, n_neighbors=15
        ).fit_transform(X)
        return coords, "UMAP"
    coords = TSNE(
        n_components=2, random_state=random_state, perplexity=30, init="pca"
    ).fit_transform(X)
    return coords, "t-SNE"


def _style_ax(ax) -> None:
    ax.set_facecolor("#F7F9FC")
    ax.grid(True, color="white", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _class_legend_handles(target_names, cmap="tab10"):
    n = max(len(target_names) - 1, 1)
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=plt.get_cmap(cmap)(i / n),
            markersize=8,
            label=name,
        )
        for i, name in enumerate(target_names)
    ]


def _noisy_knn_graph_from_L(L_noisy) -> np.ndarray:
    """Reconstruct an adjacency proxy from a (perturbed) normalized Laplacian."""
    L_dense = L_noisy.toarray() if sp.issparse(L_noisy) else np.asarray(L_noisy)
    A = -L_dense.copy()
    np.fill_diagonal(A, 0.0)
    A[A < 0] = 0.0
    return A


def _draw_graph_on_coords(
    ax,
    coords: np.ndarray,
    A,
    y: np.ndarray,
    title: str,
    edge_alpha: float = 0.15,
    edge_color: str = "gray",
    max_edges: int = 2000,
):
    A_coo = sp.triu(sp.csr_matrix(A), k=1).tocoo()
    n_edges = A_coo.nnz
    if n_edges > max_edges:
        idx = np.random.default_rng(0).choice(n_edges, size=max_edges, replace=False)
        rows, cols, vals = A_coo.row[idx], A_coo.col[idx], A_coo.data[idx]
    else:
        rows, cols, vals = A_coo.row, A_coo.col, A_coo.data
    if len(vals) > 0:
        w_norm = vals / (vals.max() + 1e-12)
        for i, j, w in zip(rows, cols, w_norm):
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=edge_color,
                alpha=edge_alpha * w,
                linewidth=0.5,
                zorder=1,
            )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=y,
        cmap="tab10",
        s=15,
        alpha=0.9,
        edgecolors="white",
        linewidth=0.3,
        zorder=2,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# ── Public visualization functions ────────────────────────────────────────────


def visualize_graph_perturbation(
    X           : np.ndarray,
    y           : np.ndarray,
    model_result: "EmbeddingResult",
    target_names: list[str] | None   = None,
    method      : str                = "umap",
    coords      : np.ndarray | None  = None,
    figsize     : tuple              = (15, 5),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side: clean k-NN graph | perturbed Laplacian | edge diff.
    Returns (fig, coords) — pass coords back to reuse the same layout.
    """
    X_sc = StandardScaler().fit_transform(X)
    if coords is None:
        coords, method_name = _reduce_2d(X_sc, method=method)
    else:
        method_name = method.upper()

    A_clean = kneighbors_graph(X_sc, n_neighbors=10, mode="connectivity", include_self=False)
    A_clean = 0.5 * (A_clean + A_clean.T)
    A_noisy = _noisy_knn_graph_from_L(model_result.L)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _draw_graph_on_coords(axes[0], coords, A_clean.toarray(), y,
                          title=f"Clean k-NN graph\n({method_name} layout)",
                          edge_color="steelblue", edge_alpha=0.25)

    scale     = getattr(model_result.noise_metadata, "scale", None)
    eps_label = f"noise scale={scale:.3e}" if isinstance(scale, float) else ""
    _draw_graph_on_coords(axes[1], coords, A_noisy, y,
                          title=f"Perturbed Laplacian\n({eps_label})",
                          edge_color="crimson", edge_alpha=0.25)

    A_clean_d = A_clean.toarray()
    diff      = A_noisy - A_clean_d
    A_removed = sp.csr_matrix(np.where(diff < -0.01, -diff, 0))
    A_added   = sp.triu(sp.csr_matrix(np.where(diff > 0.01, diff, 0)), k=1).tocoo()
    _draw_graph_on_coords(axes[2], coords, A_removed.toarray(), y,
                          title="Edge changes\n(blue=removed, red=added)",
                          edge_color="steelblue", edge_alpha=0.4)
    for i, j in zip(A_added.row, A_added.col):
        axes[2].plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                     color="crimson", alpha=0.4, linewidth=0.5, zorder=1)

    plt.tight_layout()
    return fig, coords


def visualize_embedding_shift(
    model_result: "EmbeddingResult",
    y           : np.ndarray,
    target_names: list[str] | None = None,
    method      : str              = "umap",
    figsize     : tuple            = (12, 5),
) -> plt.Figure:
    """Side-by-side: clean embedding vs noisy embedding (both in 2D)."""
    emb_clean = model_result.embedding_clean
    emb_noisy = model_result.embedding_noisy
    if emb_clean.shape[1] == 2:
        coords_c, coords_n, method_name = emb_clean, emb_noisy, "raw"
    else:
        coords_c, method_name = _reduce_2d(emb_clean, method=method, random_state=42)
        coords_n, _ = _reduce_2d(emb_noisy, method=method, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, coords, title in [
        (axes[0], coords_c, f"Clean embedding ({method_name})"),
        (axes[1], coords_n, f"Noisy embedding ({method_name})"),
    ]:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=y,
            cmap="tab10",
            s=20,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.3,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    if target_names is not None:
        fig.legend(
            handles=_class_legend_handles(target_names),
            loc="center right",
            bbox_to_anchor=(1.12, 0.5),
        )
    plt.tight_layout()
    return fig


def visualize_knn_graph(
    X: np.ndarray,
    y: np.ndarray,
    target_names: list[str] | None = None,
    n_neighbors: int = 27,
    method: str = "umap",
    coords: np.ndarray | None = None,
    figsize: tuple = (12, 10),
    max_edges_drawn: int = 3000,
    save_path: str | None = None,
) -> plt.Figure:
    """Spatial visualization of a k-NN graph with class-colored nodes."""
    X_sc = StandardScaler().fit_transform(X)
    A = kneighbors_graph(
        X_sc, n_neighbors=n_neighbors, mode="distance", include_self=False, n_jobs=-1
    )
    A = 0.5 * (A + A.T)

    if coords is None:
        coords, layout_name = _reduce_2d(X_sc, method=method)
    else:
        layout_name = method.upper()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    _style_ax(ax)

    A_coo = sp.triu(A, k=1).tocoo()
    n_edges = A_coo.nnz
    if n_edges > max_edges_drawn:
        idx = np.random.default_rng(0).choice(n_edges, max_edges_drawn, replace=False)
        rows, cols, vals = A_coo.row[idx], A_coo.col[idx], A_coo.data[idx]
        edge_note = f"(showing {max_edges_drawn:,} of {n_edges:,} edges)"
    else:
        rows, cols, vals = A_coo.row, A_coo.col, A_coo.data
        edge_note = f"({n_edges:,} edges)"

    if len(vals) > 0:
        v_max = vals.max()
        for i, j, w in zip(rows, cols, vals):
            alpha = 0.35 * (1 - w / v_max) + 0.05
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="#888888",
                alpha=alpha,
                linewidth=0.4,
                zorder=1,
            )

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=y,
        cmap="tab10",
        s=20,
        alpha=0.85,
        edgecolors="white",
        linewidth=0.3,
        zorder=2,
    )
    ax.set_title(
        f"k-NN graph (k={n_neighbors}, {layout_name}) {edge_note}", fontsize=13
    )
    ax.set_xticks([])
    ax.set_yticks([])

    if target_names is not None:
        ax.legend(
            handles=_class_legend_handles(target_names),
            loc="upper right",
            framealpha=0.9,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_privacy_utility_tradeoff(
    records         : "list[EpsilonRecord]",
    baseline_records: "list[BaselineRecord] | None" = None,
    stage_title     : str                           = "Noisy Embedding",
    figsize         : tuple                         = (10, 7.5),
    save_path       : str | None                    = None,
) -> plt.Figure:
    """
    Privacy-utility tradeoff scatter: Adversary AUC (x) vs Macro F1 (y),
    one point per epsilon value.

    The green/red separator is the mean Random Baseline (majority-class accuracy)
    averaged across the epsilon sweep, taken directly from the inference attack results.
    """
    recs_sorted     = sorted(records, key=lambda r: r.epsilon)
    adv_aucs        = np.array([r.adv_auc  for r in recs_sorted])
    f1s             = np.array([r.macro_f1 for r in recs_sorted])
    eps_vals        = np.array([r.epsilon  for r in recs_sorted])
    random_baseline = float(np.nanmean([r.random_baseline for r in recs_sorted]))

    norm = mcolors.LogNorm(vmin=eps_vals.min(), vmax=eps_vals.max())
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    _style_ax(ax)

    xlim_lo = 0.45
    xlim_hi = max(float(adv_aucs.max()) + 0.05, 0.85)
    ax.axvspan(xlim_lo, random_baseline, alpha=0.08, color="green", zorder=0)
    ax.axvspan(random_baseline, xlim_hi, alpha=0.05, color="red",   zorder=0)
    ax.axvline(random_baseline, color="#666666", linewidth=1.0, linestyle=":",
               zorder=1, label=f"Random baseline = {random_baseline:.3f}")

    ax.plot(adv_aucs, f1s, color="#5B7CB8", linewidth=1.6, alpha=0.7, zorder=2)

    for i in range(len(adv_aucs) - 1):
        dx, dy = adv_aucs[i + 1] - adv_aucs[i], f1s[i + 1] - f1s[i]
        ax.annotate(
            "",
            xy=(adv_aucs[i] + dx * 0.55, f1s[i] + dy * 0.55),
            xytext=(adv_aucs[i] + dx * 0.45, f1s[i] + dy * 0.45),
            arrowprops=dict(arrowstyle="->", color="#5B7CB8", alpha=0.5, lw=1.2),
            zorder=2,
        )

    sc = ax.scatter(
        adv_aucs,
        f1s,
        c=eps_vals,
        cmap=cmap,
        norm=norm,
        s=180,
        zorder=3,
        edgecolors="white",
        linewidths=1.5,
    )

    for i, (auc, f1, eps) in enumerate(zip(adv_aucs, f1s, eps_vals)):
        if i == 0:
            offset, ha = (10, 10), "left"
        elif i == len(adv_aucs) - 1:
            offset, ha = (10, -14), "left"
        else:
            offset, ha = ((10, 10), "left") if i % 2 == 0 else ((-12, -14), "right")
        ax.annotate(
            f"ε={eps:g}",
            xy=(auc, f1),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            color="#2C3E50",
            fontweight="bold",
            ha=ha,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#CCCCCC",
                alpha=0.85,
                linewidth=0.5,
            ),
        )

    if baseline_records:
        for br in baseline_records:
            ax.axhline(br.macro_f1, linestyle="--", linewidth=1.2, alpha=0.6,
                       label=f"{br.stage} F1={br.macro_f1:.3f}")
    ax.legend(fontsize=9, loc="lower right")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Privacy budget ε", fontsize=11)
    ax.set_xlabel("Adversary AUC (↓ = more private)", fontsize=12)
    ax.set_ylabel("Macro F1 (↑ = more useful)", fontsize=12)
    ax.set_title(
        f"Privacy–Utility Tradeoff — {stage_title}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_fiedler_evolution(
    fiedler_records: "list[FiedlerRecord]",
    figsize        : tuple        = (11, 9),
    save_path      : str | None   = None,
) -> plt.Figure:
    """Two-panel figure: (1) absolute Fiedler value vs ε, (2) ratio vs ε."""
    recs          = sorted(fiedler_records, key=lambda r: r.epsilon)
    eps_vals      = np.array([r.epsilon       for r in recs])
    fiedler_clean = np.array([r.fiedler_clean for r in recs])
    fiedler_noisy = np.array([r.fiedler_noisy for r in recs])
    fiedler_ratio = np.array([r.fiedler_ratio for r in recs])
    clean_ref     = fiedler_clean[0]

    norm = mcolors.LogNorm(vmin=eps_vals.min(), vmax=eps_vals.max())
    cmap = cm.viridis

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.32}
    )
    fig.patch.set_facecolor("white")

    # ── Panel 1: absolute values ──────────────────────────────────────────
    _style_ax(ax1)
    ax1.axhline(
        clean_ref,
        color="#2D7A2D",
        linestyle="--",
        linewidth=2,
        label=f"Clean λ₂ = {clean_ref:.4f}",
        zorder=2,
    )
    ax1.axhspan(
        clean_ref * 0.95,
        clean_ref * 1.05,
        alpha=0.12,
        color="#2D7A2D",
        zorder=1,
        label="±5% band",
    )
    ax1.plot(
        eps_vals, fiedler_noisy, color="#5B7CB8", linewidth=2.2, alpha=0.85, zorder=3
    )
    sc = ax1.scatter(
        eps_vals,
        fiedler_noisy,
        c=eps_vals,
        cmap=cmap,
        norm=norm,
        s=180,
        zorder=4,
        edgecolors="white",
        linewidths=1.5,
        label="Noisy λ₂",
    )
    for eps, noisy in zip(eps_vals, fiedler_noisy):
        ax1.plot(
            [eps, eps],
            [clean_ref, noisy],
            color="#999999",
            linewidth=0.8,
            alpha=0.5,
            linestyle=":",
            zorder=2,
        )

    for i, (eps, noisy) in enumerate(zip(eps_vals, fiedler_noisy)):
        delta_pct = 100 * (noisy - clean_ref) / clean_ref
        sign = "+" if delta_pct >= 0 else ""
        offset = (10, 12) if i % 2 == 0 else (10, -18)
        ax1.annotate(
            f"ε={eps:g}\n{sign}{delta_pct:.1f}%",
            xy=(eps, noisy),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color="#2C3E50",
            fontweight="medium",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#CCCCCC",
                alpha=0.85,
                linewidth=0.5,
            ),
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Privacy budget ε (log scale)", fontsize=12)
    ax1.set_ylabel("Fiedler value (λ₂)", fontsize=12)
    ax1.set_title(
        "Fiedler Value Drift Under Differential Privacy Noise",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=10, loc="lower right")
    plt.colorbar(sc, ax=ax1, label="ε")

    # ── Panel 2: ratio ────────────────────────────────────────────────────
    _style_ax(ax2)
    ax2.axhline(
        1.0,
        color="#2D7A2D",
        linestyle="--",
        linewidth=1.5,
        label="Ideal ratio = 1.0",
        zorder=2,
    )
    ax2.axhspan(0.95, 1.05, alpha=0.12, color="#2D7A2D", zorder=1, label="±5% band")
    ax2.plot(
        eps_vals, fiedler_ratio, color="#E07B54", linewidth=2.2, alpha=0.85, zorder=3
    )
    ax2.scatter(
        eps_vals,
        fiedler_ratio,
        c=eps_vals,
        cmap=cmap,
        norm=norm,
        s=180,
        zorder=4,
        edgecolors="white",
        linewidths=1.5,
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Privacy budget ε (log scale)", fontsize=12)
    ax2.set_ylabel("Fiedler ratio (noisy / clean)", fontsize=12)
    ax2.set_title("Fiedler Value Ratio Across ε", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
