#!/usr/bin/env python3
"""Figure 4: CFD sampling design.

Four panels: (a) SVF vs slope feature space, (b) SVF vs lambda_p,
(c) allocation bar chart by stratum, (d) per-site patch location maps
as a full row for legibility.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def main():
    apply_style()

    # Load all grids and patches
    grids = {}
    patches = {}
    for site in SITE_ORDER:
        try:
            grids[site] = load_grid(site)
            patches[site] = load_campaign_patches(site)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    # Load campaign strata summary
    strata_path = (
        PROJECT_ROOT
        / "outputs"
        / "comparative"
        / "final_allocation"
        / "campaign_strata_summary.csv"
    )
    strata_summary = (
        pd.read_csv(strata_path, index_col=0) if strata_path.exists() else None
    )

    # ── Layout: 3 rows ──
    # Row 1: feature space (a, b)
    # Row 2: bar chart (c) — full width
    # Row 3: mini-maps (d) — full width, taller
    fig = plt.figure(figsize=(WIDTH_DOUBLE, HEIGHT_MAX * 0.92))
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1, 0.85, 0.7],
        hspace=0.35,
        wspace=0.30,
    )

    # ── Panel (a): SVF vs slope ──
    ax_a = fig.add_subplot(gs[0, 0])
    for site in SITE_ORDER:
        if site not in grids:
            continue
        g = grids[site]
        valid = g[["svf", "slope_deg"]].dropna()
        ax_a.scatter(
            valid["svf"],
            valid["slope_deg"],
            s=1,
            c=SITE_COLORS[site],
            alpha=0.05,
            rasterized=True,
        )
    for site in SITE_ORDER:
        if site not in patches:
            continue
        p = patches[site]
        v = p[["svf", "slope_deg"]].dropna()
        ax_a.scatter(
            v["svf"],
            v["slope_deg"],
            s=20,
            c=SITE_COLORS[site],
            marker=SITE_MARKERS[site],
            edgecolors="k",
            linewidths=0.3,
            zorder=5,
            label=SITE_LABELS[site],
        )

    # Labeled threshold lines
    for t in SVF_THRESHOLDS:
        ax_a.axvline(t, color="#aaa", linewidth=0.3, linestyle=":")
        ax_a.text(
            t + 0.01,
            ax_a.get_ylim()[1] * 0.95,
            f"SVF={t}",
            fontsize=4.5,
            color="#888",
            va="top",
        )
    ax_a.axhline(SLOPE_THRESHOLD, color="#aaa", linewidth=0.3, linestyle=":")
    ax_a.text(
        0.95,
        SLOPE_THRESHOLD + 1,
        f"{SLOPE_THRESHOLD:.0f}\u00b0",
        fontsize=4.5,
        color="#888",
        ha="right",
    )
    ax_a.set_xlabel(r"$SVF$")
    ax_a.set_ylabel("Slope (\u00b0)")
    ax_a.legend(loc="upper right", frameon=False, markerscale=1.5, fontsize=5)

    # ── Panel (b): SVF vs λp ──
    ax_b = fig.add_subplot(gs[0, 1])
    for site in SITE_ORDER:
        if site not in grids:
            continue
        g = grids[site]
        valid = g[["svf", "lambda_p"]].dropna()
        ax_b.scatter(
            valid["svf"],
            valid["lambda_p"],
            s=1,
            c=SITE_COLORS[site],
            alpha=0.05,
            rasterized=True,
        )
    for site in SITE_ORDER:
        if site not in patches:
            continue
        p = patches[site]
        v = p[["svf", "lambda_p"]].dropna()
        ax_b.scatter(
            v["svf"],
            v["lambda_p"],
            s=20,
            c=SITE_COLORS[site],
            marker=SITE_MARKERS[site],
            edgecolors="k",
            linewidths=0.3,
            zorder=5,
        )
    for t in SVF_THRESHOLDS:
        ax_b.axvline(t, color="#aaa", linewidth=0.3, linestyle=":")
    ax_b.axhline(LAMBDA_P_THRESHOLD, color="#aaa", linewidth=0.3, linestyle=":")
    ax_b.text(
        0.95,
        LAMBDA_P_THRESHOLD + 0.02,
        f"$\\lambda_p$={LAMBDA_P_THRESHOLD}",
        fontsize=4.5,
        color="#888",
        ha="right",
    )
    ax_b.set_xlabel(r"$SVF$")
    ax_b.set_ylabel(r"$\lambda_p$")

    # ── Panel (c): allocation bar chart (full width) ──
    ax_c = fig.add_subplot(gs[1, :])
    if strata_summary is not None:
        strata_ids = sorted(strata_summary.index)
        y = np.arange(len(strata_ids))
        lefts = np.zeros(len(strata_ids))

        for site in SITE_ORDER:
            if site not in strata_summary.columns:
                continue
            vals = [
                strata_summary.loc[sid, site] if sid in strata_summary.index else 0
                for sid in strata_ids
            ]
            ax_c.barh(
                y,
                vals,
                left=lefts,
                height=0.65,
                color=SITE_COLORS[site],
                edgecolor="white",
                linewidth=0.3,
                label=SITE_LABELS[site],
            )
            lefts += np.array(vals)

        # Total annotations at end of each bar
        for i, sid in enumerate(strata_ids):
            total = int(lefts[i])
            if total > 0:
                ax_c.text(
                    total + 0.3,
                    i,
                    str(total),
                    va="center",
                    fontsize=5.5,
                    fontweight="bold",
                )
            else:
                ax_c.text(0.3, i, "\u2014", va="center", fontsize=5, color="#999")

        ax_c.set_yticks(y)
        ax_c.set_yticklabels(strata_ids, fontsize=5.5)
        ax_c.set_xlabel("Patches")
        ax_c.invert_yaxis()
        ax_c.legend(loc="lower right", frameon=False, fontsize=5.5, ncol=5)

    # ── Panel (d): mini-maps (full row, one per site) ──
    gs_d = gs[2, :].subgridspec(1, 5, wspace=0.08)

    for j, site in enumerate(SITE_ORDER):
        ax = fig.add_subplot(gs_d[0, j])

        try:
            bld = load_buildings(site)
            bld.plot(
                ax=ax, facecolor="#e0e0e0", edgecolor="#999", linewidth=0.05, alpha=0.6
            )
        except Exception:
            pass

        if site in patches:
            p = patches[site]
            for _, row in p.iterrows():
                cx = row.get("center_x", row.get("centroid_x"))
                cy = row.get("center_y", row.get("centroid_y"))
                ax.add_patch(
                    plt.Circle(
                        (cx, cy),
                        50,
                        linewidth=0.6,
                        edgecolor="k",
                        facecolor=SITE_COLORS[site],
                        alpha=0.4,
                        zorder=5,
                    )
                )

        clean_map_axes(ax)
        ax.set_aspect("equal")
        n = len(patches[site]) if site in patches else 0
        ax.set_title(f"{SITE_LABELS[site]} (n={n})", fontsize=5.5, pad=2)
        add_scalebar(ax, length_m=200)

    save_fig(fig, "fig04_sampling_design")


if __name__ == "__main__":
    main()
