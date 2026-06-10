#!/usr/bin/env python3
"""BRISA slide deck — fig_vidigal_winter_solar_v2.png.

Slide-13 hero map. Vidigal winter-solstice street-level direct-sun hours.
Designed for projection: 16:9 aspect, map fills frame, oversized colorbar
and scalebar, no UTM axis labels, no in-figure title or stats box (slide
adds those itself).

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_vidigal_winter_solar_v2.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
sys.path.insert(0, str(PROJECT_ROOT))
from fig_style import load_buildings  # noqa: E402
from src.viz import presentation_style as ps  # noqa: E402

PAPER_OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
PRESENTATION_OUT_DIR = PROJECT_ROOT / "outputs" / "vidigal" / "presentation_figures"
OUT_FILENAME = "fig_vidigal_winter_solar_v2.png"

SOLAR_PATH = (
    PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
)

VMAX_HOURS = 12.0
CMAP = "YlOrRd_r"  # 0 h = deep red (worst), 12 h = pale yellow (best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=("paper", "presentation"), default="paper")
    args = parser.parse_args()

    ps.apply(args.preset)
    pts = gpd.read_file(SOLAR_PATH)
    buildings = load_buildings("vidigal", extended=False)
    if buildings.crs != pts.crs:
        buildings = buildings.to_crs(pts.crs)

    # 16:9 figure, 12 in × 6.75 in @ 300 dpi → 3600 × 2025 px
    fig_w, fig_h = 12.0, 6.75
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    # Map fills nearly the full canvas; thin right strip for colorbar
    ax = fig.add_axes([0.005, 0.02, 0.86, 0.96])
    cax = fig.add_axes([0.885, 0.10, 0.025, 0.80])

    buildings.plot(ax=ax, facecolor="#e3ddd2", edgecolor="#9a9388",
                   linewidth=0.25, zorder=1)
    sc = ax.scatter(
        pts.geometry.x, pts.geometry.y,
        c=pts["solar_hours_winter"],
        cmap=CMAP, vmin=0, vmax=VMAX_HOURS,
        s=14, alpha=0.95, linewidths=0, zorder=2,
    )

    bbox = pts.total_bounds
    pad_x = (bbox[2] - bbox[0]) * 0.02
    pad_y = (bbox[3] - bbox[1]) * 0.02
    ax.set_xlim(bbox[0] - pad_x, bbox[2] + pad_x)
    ax.set_ylim(bbox[1] - pad_y, bbox[3] + pad_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar — projection-scale fonts
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("winter direct-sun (h)", fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=13)
    cbar.outline.set_linewidth(0.6)

    # Scalebar — bottom-left, large label
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ex = xlim[1] - xlim[0]
    ey = ylim[1] - ylim[0]
    bar_len_m = 200.0
    bar_h = ey * 0.010
    x0 = xlim[0] + ex * 0.04
    y0 = ylim[0] + ey * 0.06
    ax.add_patch(plt.Rectangle(
        (x0, y0), bar_len_m, bar_h,
        facecolor="#111111", edgecolor="#111111", linewidth=0.6, zorder=20,
    ))
    ax.text(
        x0 + bar_len_m / 2, y0 + bar_h + ey * 0.012,
        f"{bar_len_m:.0f} m",
        ha="center", va="bottom", fontsize=18, fontweight="bold",
        color="#111111", zorder=20,
    )

    # North arrow — upper right of map
    arrow_x = xlim[1] - ex * 0.05
    arrow_y0 = ylim[1] - ey * 0.18
    arrow_y1 = ylim[1] - ey * 0.08
    ax.annotate(
        "", xy=(arrow_x, arrow_y1), xytext=(arrow_x, arrow_y0),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=2.2,
                        mutation_scale=22),
        zorder=20,
    )
    ax.text(
        arrow_x, arrow_y1 + ey * 0.015, "N",
        ha="center", va="bottom", fontsize=18, fontweight="bold",
        color="#111111", zorder=20,
    )

    out_dir = PRESENTATION_OUT_DIR if args.preset == "presentation" else PAPER_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / OUT_FILENAME
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    size_kb = out_png.stat().st_size // 1024
    print(f"  Saved {out_png} ({size_kb} KB, {fig_w:.1f}x{fig_h:.2f} in @ 300 dpi)")


if __name__ == "__main__":
    main()
