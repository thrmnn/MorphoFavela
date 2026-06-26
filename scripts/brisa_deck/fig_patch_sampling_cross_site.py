#!/usr/bin/env python3
"""BRISA slide deck — fig_patch_sampling_cross_site.png (supplementary S15).

Small-multiples grid: the patch-sampling stratification visualised per site
across all 5 favelas. Same aesthetic as fig_patch_sampling_vidigal_v2.png but
lower-resolution per panel.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_patch_sampling_cross_site.png
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
logging.getLogger("pilot_sampling").setLevel(logging.ERROR)
from scripts.run_pilot_sampling import (
    CONFIG,
    PATCH_RADIUS_M,
    SITE_PRESETS,
    _hillshade,
    _resolve,
    assign_strata,
)

PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
from _assets import BRISA_ASSETS_DIR
from fig_style import SITE_LABELS, apply_style

OUT_DIR = BRISA_ASSETS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

# Use a fixed colour-per-stratum mapping so colours mean the same thing across
# all 5 panels (the per-site _stratum_cmap helper assigns colours by the order
# of populated strata, which would re-use the same colour for different strata
# in different sites — confusing for a cross-site comparison).
ALL_STRATA = [
    "SVF1_SLP1_LP1", "SVF1_SLP1_LP2",
    "SVF1_SLP2_LP1", "SVF1_SLP2_LP2",
    "SVF2_SLP1_LP1", "SVF2_SLP1_LP2",
    "SVF2_SLP2_LP1", "SVF2_SLP2_LP2",
    "SVF3_SLP1_LP1", "SVF3_SLP1_LP2",
    "SVF3_SLP2_LP1", "SVF3_SLP2_LP2",
]
STRATUM_COLORS = {
    "SVF1_SLP1_LP1": "#e6194b", "SVF1_SLP1_LP2": "#3cb44b",
    "SVF1_SLP2_LP1": "#ffe119", "SVF1_SLP2_LP2": "#4363d8",
    "SVF2_SLP1_LP1": "#f58231", "SVF2_SLP1_LP2": "#911eb4",
    "SVF2_SLP2_LP1": "#42d4f4", "SVF2_SLP2_LP2": "#f032e6",
    "SVF3_SLP1_LP1": "#bfef45", "SVF3_SLP1_LP2": "#fabed4",
    "SVF3_SLP2_LP1": "#469990", "SVF3_SLP2_LP2": "#dcbeff",
}


def stratum_label(sid: str) -> str:
    parts = sid.split("_")
    svf_map = {"SVF1": "SVF<0.15", "SVF2": "SVF 0.15–0.30", "SVF3": "SVF≥0.30"}
    slp_map = {"SLP1": "slope<15°", "SLP2": "slope≥15°"}
    lp_map = {"LP1": "λp<0.5", "LP2": "λp≥0.5"}
    return f"{svf_map.get(parts[0])} · {slp_map.get(parts[1])} · {lp_map.get(parts[2])}"


def site_config(site: str) -> dict:
    cfg = dict(CONFIG)
    cfg.update(SITE_PRESETS[site])
    return cfg


def render_panel(ax, site: str) -> int:
    """Render one site panel, return number of patches drawn."""
    cfg = site_config(site)
    grid = gpd.read_file(_resolve(cfg["grid_metrics"]))
    grid = assign_strata(grid, cfg)

    patch_path = (
        PROJECT_ROOT / "outputs" / cfg["site_id"]
        / "sampling_cfd" / "campaign_sampling" / "campaign_patches.csv"
    )
    patches = pd.read_csv(patch_path) if patch_path.exists() else pd.DataFrame()

    dtm_path = _resolve(cfg["dtm"])
    if dtm_path.exists():
        try:
            with rasterio.open(dtm_path) as src:
                dtm = src.read(1).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    dtm[dtm == nodata] = np.nan
                bounds = src.bounds
                shade = _hillshade(dtm, src.res[0])
                shade = np.where(np.isnan(dtm), 1.0, shade)
                ax.imshow(
                    shade, cmap="gray", vmin=0, vmax=1,
                    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                    alpha=0.35, zorder=0,
                )
        except Exception:
            pass

    g = grid[grid["stratum_id"].notna()].copy()
    g["_color"] = g["stratum_id"].map(STRATUM_COLORS).fillna("#cccccc")
    g.plot(ax=ax, color=g["_color"], alpha=0.55, edgecolor="none", zorder=1)

    for _, row in patches.iterrows():
        cx, cy = row["center_x"], row["center_y"]
        ax.add_patch(
            plt.Circle(
                (cx, cy), PATCH_RADIUS_M,
                linewidth=0.9, edgecolor="black", facecolor="none", zorder=5,
            )
        )
        ax.plot(cx, cy, marker="o", color="black", markersize=2.0, zorder=6)

    gbb = g.total_bounds
    pad = max((gbb[2] - gbb[0]), (gbb[3] - gbb[1])) * 0.04
    ax.set_xlim(gbb[0] - pad, gbb[2] + pad)
    ax.set_ylim(gbb[1] - pad, gbb[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Per-panel 200 m scalebar.
    x0 = gbb[0] - pad + (gbb[2] - gbb[0]) * 0.04
    y0 = gbb[1] - pad + (gbb[3] - gbb[1]) * 0.05
    ax.add_patch(
        plt.Rectangle(
            (x0, y0), 200.0, (gbb[3] - gbb[1]) * 0.012,
            facecolor="black", edgecolor="black", linewidth=0,
            zorder=20, clip_on=False,
        )
    )
    ax.text(
        x0 + 100, y0 + (gbb[3] - gbb[1]) * 0.030,
        "200 m", ha="center", va="bottom", fontsize=7, color="#111111",
    )

    n_patches = len(patches)
    ax.set_title(f"{SITE_LABELS[site]}  (n={n_patches} patches)",
                 fontsize=10, pad=4, color="#111111")
    return n_patches


def main() -> None:
    apply_style()

    fig_w = 18.0
    fig_h = fig_w * 0.55  # ~10 in tall
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Layout: 5 panels in a row, each sized to its own bbox aspect.
    bboxes = {}
    for s in SITES:
        cfg = site_config(s)
        gpath = _resolve(cfg["grid_metrics"])
        g = gpd.read_file(gpath)
        g = assign_strata(g, cfg)
        gv = g[g["stratum_id"].notna()]
        bboxes[s] = gv.total_bounds
    aspects = []
    for s in SITES:
        b = bboxes[s]
        aspects.append((b[2] - b[0]) / (b[3] - b[1]))

    left, right = 0.02, 0.985
    top, bottom = 0.95, 0.22
    band_w = right - left
    band_h = top - bottom
    inter = 0.012

    # Per-site panel sized so the LONGER dimension of every site fills the same
    # display length. Wide sites (Vidigal) get short panels; tall sites (Maré)
    # get narrow panels but the full band height. Centred vertically in the
    # band so panels share a common midline.
    fig_aspect = fig_h / fig_w  # data-fraction units
    # Pick a per-site "max display length" in figure-width fraction. We want
    # the sum of panel widths to fit band_w; pick max length so that the
    # widest site exactly uses its naturally implied width.
    band_w_eff = band_w - inter * (len(SITES) - 1)
    # Each site's natural width given a unit max-length: width_unit = min(aspect, 1)
    # height_unit = min(1/aspect, 1). All max-dimensions equal 1.
    width_units = np.array([min(a, 1.0) for a in aspects])
    height_units = np.array([min(1.0 / a, 1.0) for a in aspects])
    # Solve: max_length * sum(width_units) = band_w_eff
    max_length = band_w_eff / width_units.sum()
    panel_widths = max_length * width_units
    panel_heights_fig = max_length * height_units * (fig_w / fig_h)
    # If any panel exceeds the band height, shrink everything.
    if panel_heights_fig.max() > band_h:
        scale = band_h / panel_heights_fig.max()
        max_length *= scale
        panel_widths = max_length * width_units
        panel_heights_fig = max_length * height_units * (fig_w / fig_h)

    x = left
    mid_y = bottom + band_h / 2
    for i, s in enumerate(SITES):
        py = mid_y - panel_heights_fig[i] / 2
        ax = fig.add_axes([x, py, panel_widths[i], panel_heights_fig[i]])
        render_panel(ax, s)
        x += panel_widths[i] + inter

    # Shared bottom legend.
    stratum_handles = [
        Patch(facecolor=STRATUM_COLORS[sid], alpha=0.85, edgecolor="none",
              label=stratum_label(sid))
        for sid in ALL_STRATA
    ]
    patch_handle = Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="white", markeredgecolor="black",
        markersize=8, markeredgewidth=1.0,
        label="CFD patch (100 m diameter)",
    )
    leg = fig.legend(
        handles=stratum_handles + [patch_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        fontsize=9,
        title="Morphometric clusters in joint SVF × λp × slope feature space",
        title_fontsize=10,
        labelspacing=0.45,
        handletextpad=0.6,
        columnspacing=1.4,
    )
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#888888")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)

    out = OUT_DIR / "fig_patch_sampling_cross_site.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
