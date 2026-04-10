#!/usr/bin/env python3
"""Figure 2: Morphometric distributions across sites.

Five-panel violin plot showing SVF, lambda_p, lambda_f, sigma_H, and slope
distributions for all 5 sites, ordered by typology with grouping brackets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

import matplotlib.pyplot as plt
import numpy as np


def main():
    apply_style()

    indicators = [
        ("svf",           r"$SVF$"),
        ("lambda_p",      r"$\lambda_p$"),
        ("lambda_f_mean", r"$\lambda_f$"),
        ("sigma_h",       r"$\sigma_H$ (m)"),
        ("slope_deg",     "Slope (\u00b0)"),
    ]

    thresholds = {
        "svf": SVF_THRESHOLDS,
        "slope_deg": [SLOPE_THRESHOLD],
        "lambda_p": [LAMBDA_P_THRESHOLD],
    }

    # Load all grids
    grids = {}
    for site in SITE_ORDER:
        try:
            grids[site] = load_grid(site)
        except FileNotFoundError:
            print(f"  WARNING: grid not found for {site}")

    # 1 row x 5 panels (wider, no porosity)
    fig, axes = plt.subplots(1, 5, figsize=(WIDTH_DOUBLE, WIDTH_DOUBLE * 0.30))

    # Typology background shading
    typology_colors = {"hillside": "#CC667710", "flatland": "#44AA9910"}

    for ax, (col, label) in zip(axes, indicators):
        data = []
        colors = []
        positions = []

        for i, site in enumerate(SITE_ORDER):
            if site not in grids:
                continue
            vals = grids[site][col].dropna().values
            if len(vals) == 0:
                continue
            data.append(vals)
            colors.append(SITE_COLORS[site])
            positions.append(i)

        if not data:
            ax.set_visible(False)
            continue

        # Typology background shading
        ax.axvspan(-0.5, 1.5, color="#CC6677", alpha=0.04, zorder=0)   # hillside
        ax.axvspan(3.5, 4.5, color="#44AA99", alpha=0.04, zorder=0)    # flatland

        parts = ax.violinplot(
            data, positions=positions, showextrema=False, showmedians=False,
            widths=0.75,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.65)
            pc.set_edgecolor(colors[i])
            pc.set_linewidth(0.3)

        # Median + IQR
        for i, (vals, pos) in enumerate(zip(data, positions)):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.vlines(pos, q1, q3, color="k", linewidth=1.0, zorder=5)
            ax.scatter(pos, med, color="white", s=10, zorder=6,
                       edgecolors="k", linewidths=0.4)

        # Threshold lines
        if col in thresholds:
            for t in thresholds[col]:
                ax.axhline(t, color="#888888", linewidth=0.4, linestyle="--", zorder=1)

        ax.set_xticks(range(len(SITE_ORDER)))
        ax.set_xticklabels(
            [SITE_LABELS.get(s, s) for s in SITE_ORDER],
            rotation=35, ha="right", fontsize=6,
        )
        ax.set_ylabel(label, fontsize=7)

        # n annotations
        for i, (vals, pos) in enumerate(zip(data, positions)):
            ax.text(pos, ax.get_ylim()[1] * 0.97, f"n={len(vals):,d}",
                    ha="center", va="top", fontsize=4, color="#999999")

    # Typology bracket annotations on the first panel
    ax0 = axes[0]
    ylim = ax0.get_ylim()
    y_bracket = ylim[1] * 1.08

    fig.tight_layout(w_pad=0.8)
    save_fig(fig, "fig02_morphometric_distributions")


if __name__ == "__main__":
    main()
