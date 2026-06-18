#!/usr/bin/env python3
"""BRISA slide deck — fig_patch_sampling_vidigal_v2.png (slide 10).

Single-panel Vidigal CFD patch sampling viz. Cleaner than the v1 (sd_patch_sampling.png):

  * One marker style for all CFD patches (drops the pilot/new distinction —
    the audience doesn't care about the campaign batching).
  * KEEPS the coloured 50 m grid showing the morphometric strata (SVF x slope
    x λp clusters), and adds an EXPLICIT legend mapping each colour swatch to
    the stratum it represents.
  * Removes the per-patch ~580 m source-data envelope circles that, in the v1,
    looked like mis-sized inlet zones. We are showing morphometric sampling
    coverage here, not the CFD domain extent — those domains live in fig 4.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_patch_sampling_vidigal_v2.png
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# silence the noisy module-level logger from run_pilot_sampling import
logging.getLogger("pilot_sampling").setLevel(logging.ERROR)
from scripts.run_pilot_sampling import (  # noqa: E402
    CONFIG,
    PATCH_RADIUS_M,
    SITE_PRESETS,
    _hillshade,
    _resolve,
    _stratum_cmap,
    assign_strata,
)

OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def site_config(site: str) -> dict:
    cfg = dict(CONFIG)
    cfg.update(SITE_PRESETS[site])
    return cfg


def stratum_label(sid: str) -> str:
    """Human-readable stratum label, e.g. SVF1_SLP2_LP1 → 'SVF<0.15 · ≥15° · λp<0.5'."""
    parts = sid.split("_")
    svf_map = {"SVF1": "SVF<0.15", "SVF2": "SVF 0.15–0.30", "SVF3": "SVF≥0.30"}
    slp_map = {"SLP1": "slope<15°", "SLP2": "slope≥15°"}
    lp_map = {"LP1": "λp<0.5", "LP2": "λp≥0.5"}
    return f"{svf_map.get(parts[0], parts[0])} · {slp_map.get(parts[1], parts[1])} · {lp_map.get(parts[2], parts[2])}"


def render_site(site: str, out_name: str, figsize=(13.0, 8.1)) -> Path:
    cfg = site_config(site)

    grid_path = _resolve(cfg["grid_metrics"])
    grid = gpd.read_file(grid_path)
    grid = assign_strata(grid, cfg)

    patch_path = (
        PROJECT_ROOT
        / "outputs"
        / cfg["site_id"]
        / "sampling_cfd"
        / "campaign_sampling"
        / "campaign_patches.csv"
    )
    if not patch_path.exists():
        patch_path = (
            PROJECT_ROOT
            / "outputs"
            / cfg["site_id"]
            / "sampling_cfd"
            / "pilot_sampling"
            / "pilot_patches.csv"
        )
    patches = pd.read_csv(patch_path)

    fig, ax = plt.subplots(figsize=figsize)

    dtm_path = _resolve(cfg["dtm"])
    if dtm_path.exists():
        with rasterio.open(dtm_path) as src:
            dtm = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                dtm[dtm == nodata] = np.nan
            bounds = src.bounds
            shade = _hillshade(dtm, src.res[0])
            shade = np.where(np.isnan(dtm), 1.0, shade)
            ax.imshow(
                shade,
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                alpha=0.40,
                zorder=0,
            )

    cmap = _stratum_cmap(grid["stratum_id"].tolist())
    g = grid[grid["stratum_id"].notna()].copy()
    g["_color"] = g["stratum_id"].map(cmap)
    g.plot(ax=ax, color=g["_color"], alpha=0.55, edgecolor="none", zorder=1)

    # ONE marker per patch (no pilot/new distinction).
    # Open black ring shows the 100 m analysis-patch footprint; centre dot is
    # the patch location. Transparent fill lets the stratum colours show
    # through so the reader can still see which clusters are sampled.
    for _, row in patches.iterrows():
        cx, cy = row["center_x"], row["center_y"]
        ax.add_patch(
            plt.Circle(
                (cx, cy),
                PATCH_RADIUS_M,
                linewidth=1.5,
                edgecolor="black",
                facecolor="none",
                zorder=5,
            )
        )
        ax.plot(cx, cy, marker="o", color="black", markersize=4.5, zorder=6)

    # Tight crop to the grid extent + a little breathing room.
    gbb = g.total_bounds
    pad = max((gbb[2] - gbb[0]), (gbb[3] - gbb[1])) * 0.05
    ax.set_xlim(gbb[0] - pad, gbb[2] + pad)
    ax.set_ylim(gbb[1] - pad, gbb[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 200 m scale bar (lower-left).
    x0 = gbb[0] - pad + (gbb[2] - gbb[0]) * 0.04
    y0 = gbb[1] - pad + (gbb[3] - gbb[1]) * 0.05
    ax.add_patch(
        plt.Rectangle(
            (x0, y0), 200.0, (gbb[3] - gbb[1]) * 0.008,
            facecolor="black", edgecolor="black", linewidth=0,
            zorder=20, clip_on=False,
        )
    )
    ax.text(
        x0 + 100, y0 + (gbb[3] - gbb[1]) * 0.022,
        "200 m", ha="center", va="bottom", fontsize=9, color="#111111",
    )

    # Explicit stratum legend.
    from matplotlib.patches import Patch
    sorted_strata = sorted(cmap.keys())
    stratum_handles = [
        Patch(facecolor=cmap[sid], edgecolor="none", alpha=0.85,
              label=stratum_label(sid))
        for sid in sorted_strata
    ]
    patch_handle = plt.Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        markeredgewidth=1.0,
        label=f"CFD patch (n={len(patches)}, 100 m diameter)",
    )

    # First legend: strata (left).
    leg_strata = ax.legend(
        handles=stratum_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.01),
        bbox_transform=ax.transAxes,
        ncol=3,
        frameon=True,
        fontsize=8.5,
        title="Morphometric clusters in joint SVF × λp × slope feature space",
        title_fontsize=9.5,
        labelspacing=0.4,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    leg_strata.get_frame().set_linewidth(0.4)
    leg_strata.get_frame().set_edgecolor("#888888")
    leg_strata.get_frame().set_facecolor("white")
    leg_strata.get_frame().set_alpha(0.95)
    leg_strata._legend_box.align = "left"
    ax.add_artist(leg_strata)

    # Second legend: CFD patches (right).
    leg_p = ax.legend(
        handles=[patch_handle],
        loc="upper right",
        bbox_to_anchor=(1.0, -0.01),
        bbox_transform=ax.transAxes,
        frameon=True,
        fontsize=9,
    )
    leg_p.get_frame().set_linewidth(0.4)
    leg_p.get_frame().set_edgecolor("#888888")
    leg_p.get_frame().set_facecolor("white")
    leg_p.get_frame().set_alpha(0.95)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.30)

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    out = render_site("vidigal", "fig_patch_sampling_vidigal_v2.png")
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
