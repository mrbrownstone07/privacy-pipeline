import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

features_df = pd.read_csv("features_raw_0_overlap.csv")

def plot_full_data_distribution(df, figsize=(18, 6), save_path=None):
    """Three-panel view: by fault_type, by fault_size, by load.
    Skips panels for any columns that aren't present in df."""
    candidate_panels = [
        ("fault_type", "Fault Class",    plt.cm.viridis),
        ("fault_size", "Fault Size",     plt.cm.plasma),
        ("load",       "Operating Load", plt.cm.cividis),
    ]

    # Filter to only columns that exist
    panels = [(c, l, cm) for c, l, cm in candidate_panels if c in df.columns]
    missing = [c for c, _, _ in candidate_panels if c not in df.columns]

    if not panels:
        raise ValueError(
            f"None of the expected metadata columns are in the DataFrame.\n"
            f"Expected one of: {[c for c, _, _ in candidate_panels]}\n"
            f"Available columns (first 20): {df.columns.tolist()[:20]}"
        )
    if missing:
        print(f"[warn] Skipping panels for missing columns: {missing}")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("white")

    for ax, (col, label, cmap) in zip(axes, panels):
        counts = df[col].value_counts().sort_index()
        colors = cmap(np.linspace(0.15, 0.85, len(counts)))

        ax.set_facecolor("#F7F9FC")
        bars = ax.bar(
            counts.index.astype(str), counts.values,
            color=colors, edgecolor="white", linewidth=1.2, zorder=3,
        )
        for bar, c in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:,}", ha="center", va="bottom",
                fontsize=9, color="#2C3E50",
            )

        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Segments", fontsize=10)
        ax.set_ylim(0, counts.max() * 1.15)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=30 if counts.shape[0] > 4 else 0)
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

    plt.suptitle(
        f"CWRU Dataset Distribution · {len(df):,} total segments",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


plot_full_data_distribution(features_df, save_path="cumulative_data_distribution.png")
