#!/usr/bin/env python3
"""Figure S1: Correlation matrices for all 5 sites.

Five heatmaps showing pairwise Pearson correlations between morphometric
indicators, with consistent ordering and color scale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from fig_style import *


def main():
    apply_style()

    indicators = [
        ("svf", r"$SVF$"),
        ("lambda_p", r"$\lambda_p$"),
        ("lambda_f_mean", r"$\lambda_f$"),
        ("porosity", "Porosity"),
        ("sigma_h", r"$\sigma_H$"),
        ("H_mean", r"$H_{mean}$"),
        ("slope_deg", "Slope"),
        ("street_orientation_entropy", "Entropy"),
    ]
    cols = [c for c, _ in indicators]
    labels = [l for _, l in indicators]

    fig, axes = plt.subplots(1, 5, figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.22))

    vmin, vmax = -1, 1

    for ax, site in zip(axes, SITE_ORDER):
        try:
            grid = load_grid(site)
        except FileNotFoundError:
            ax.set_visible(False)
            continue

        available = [c for c in cols if c in grid.columns]
        available_labels = [labels[cols.index(c)] for c in available]

        corr = grid[available].corr()

        im = ax.imshow(
            corr.values,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Annotate cells
        n = len(available)
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=3.5,
                    color=color,
                )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(available_labels, rotation=45, ha="right", fontsize=4.5)
        ax.set_yticklabels(available_labels if ax == axes[0] else [], fontsize=4.5)
        ax.set_title(SITE_LABELS[site], fontsize=6, pad=3)

        # Highlight strong correlations
        for i in range(n):
            for j in range(n):
                if i != j and abs(corr.values[i, j]) > 0.7:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="k",
                            linewidth=0.5,
                        )
                    )

    fig.tight_layout(w_pad=0.3)
    save_fig(fig, "figS1_correlation_matrices")


if __name__ == "__main__":
    main()
