#!/usr/bin/env python3
"""BRISA slide deck — fig_morpho_typology_scatter.png (slide 8).

Bivariate scatter of every 10 m grid cell across all 5 favelas in SVF (x) x
slope (y) feature space, coloured by hillside/flatland typology. Tens of
thousands of cells, hence very small dots with alpha < 0.3. Marginal KDEs on
each axis show that hillside vs flatland favelas occupy genuinely different
parts of the feature space — the analytic claim of slide 8.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_morpho_typology_scatter.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
from fig_style import SITE_LABELS, apply_style  # noqa: E402

OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Typology assignment matches src/typology.py; complexo_do_alemao is "mixed"
# but for the binary hillside-vs-flatland claim it groups with hillside (it
# has the same morphometric signature: steep, low-SVF cores).
HILLSIDE = {"vidigal", "rocinha", "complexo_do_alemao"}
FLATLAND = {"riodaspedras", "maré"}

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

COLOR_HILLSIDE = "#111111"
COLOR_FLATLAND = "#888888"


def load_grid(site: str) -> gpd.GeoDataFrame:
    p = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    grid = gpd.read_file(p)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    return grid[built].copy()


def main() -> None:
    apply_style()

    cells = []
    for s in SITES:
        g = load_grid(s)
        df = g[["svf", "slope_deg"]].dropna().copy()
        df["site"] = s
        df["typology"] = "hillside" if s in HILLSIDE else "flatland"
        cells.append(df)
    all_cells = (
        gpd.pd.concat(cells, ignore_index=True)
        if False
        else __import__("pandas").concat(cells, ignore_index=True)
    )
    print(f"Total cells: {len(all_cells)}")
    print(all_cells.groupby("typology").size())

    fig_w = 9.5
    fig_h = fig_w / 1.4
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Main scatter + marginal KDEs (Seaborn-style jointplot layout, hand-built).
    left, right = 0.10, 0.95
    bottom, top = 0.10, 0.92
    main_w = (right - left) * 0.78
    marg_w = (right - left) - main_w - 0.012
    main_h = (top - bottom) * 0.78
    marg_h = (top - bottom) - main_h - 0.012

    ax_main = fig.add_axes([left, bottom, main_w, main_h])
    ax_top = fig.add_axes([left, bottom + main_h + 0.012, main_w, marg_h], sharex=ax_main)
    ax_right = fig.add_axes([left + main_w + 0.012, bottom, marg_w, main_h], sharey=ax_main)

    # Scatter (flatland behind, hillside on top so hillside reads as foreground).
    rng = np.random.default_rng(42)
    for typ, color in [("flatland", COLOR_FLATLAND), ("hillside", COLOR_HILLSIDE)]:
        sub = all_cells[all_cells["typology"] == typ]
        # subsample if huge (>30k) for plot performance
        if len(sub) > 30000:
            idx = rng.choice(len(sub), 30000, replace=False)
            sub = sub.iloc[idx]
        ax_main.scatter(
            sub["svf"],
            sub["slope_deg"],
            s=2.2,
            c=color,
            alpha=0.18,
            linewidths=0,
            rasterized=True,
            label=typ,
        )

    ax_main.set_xlim(0.0, 1.0)
    ax_main.set_ylim(0.0, 65.0)
    ax_main.set_xlabel("Sky View Factor", fontsize=10)
    ax_main.set_ylabel("Slope (degrees)", fontsize=10)
    ax_main.tick_params(axis="both", labelsize=9)
    for spine in ("top", "right"):
        ax_main.spines[spine].set_visible(False)
    ax_main.spines["left"].set_linewidth(0.5)
    ax_main.spines["bottom"].set_linewidth(0.5)
    # Faint threshold reference: 15° slope (hillside/flatland convention).
    ax_main.axhline(15.0, color="#bbbbbb", lw=0.6, ls="--", zorder=0)
    ax_main.text(0.985, 15.5, "15° slope", fontsize=7.5, color="#888888",
                 ha="right", va="bottom")

    # Marginal KDEs.
    xs = np.linspace(0.0, 1.0, 400)
    ys = np.linspace(0.0, 65.0, 400)
    for typ, color in [("hillside", COLOR_HILLSIDE), ("flatland", COLOR_FLATLAND)]:
        sub = all_cells[all_cells["typology"] == typ]
        # SVF marginal (top)
        if len(sub) > 20:
            kx = gaussian_kde(sub["svf"], bw_method=0.18)
            dx = kx(xs)
            ax_top.fill_between(xs, 0, dx, color=color, alpha=0.30, linewidth=0)
            ax_top.plot(xs, dx, color=color, lw=0.9)
        # slope marginal (right)
        if len(sub) > 20:
            ky = gaussian_kde(sub["slope_deg"], bw_method=0.18)
            dy = ky(ys)
            ax_right.fill_betweenx(ys, 0, dy, color=color, alpha=0.30, linewidth=0)
            ax_right.plot(dy, ys, color=color, lw=0.9)

    for ax in (ax_top, ax_right):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Inset legend.
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_HILLSIDE,
               markersize=6, markeredgewidth=0,
               label=f"hillside ({', '.join(SITE_LABELS[s] for s in SITES if s in HILLSIDE)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_FLATLAND,
               markersize=6, markeredgewidth=0,
               label=f"flatland ({', '.join(SITE_LABELS[s] for s in SITES if s in FLATLAND)})"),
    ]
    leg = ax_main.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=8.5,
        title="Typology",
        title_fontsize=9,
    )
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#888888")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.9)

    # Annotate cell counts.
    n_h = int((all_cells["typology"] == "hillside").sum())
    n_f = int((all_cells["typology"] == "flatland").sum())
    ax_main.text(
        0.015, 0.985,
        f"n = {len(all_cells):,} cells\n  hillside: {n_h:,}\n  flatland: {n_f:,}",
        transform=ax_main.transAxes,
        ha="left", va="top",
        fontsize=8, color="#222222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.4, alpha=0.9),
    )

    out = OUT_DIR / "fig_morpho_typology_scatter.png"
    fig.savefig(out, dpi=220, facecolor="white", bbox_inches=None)
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
