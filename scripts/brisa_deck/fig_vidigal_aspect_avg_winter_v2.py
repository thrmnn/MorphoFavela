#!/usr/bin/env python3
"""BRISA slide deck — fig_vidigal_aspect_avg_winter_v2.png.

Slide-13 companion to the winter solar map. Same data and palette as v1,
re-tuned for projection: 16:9 aspect at 300 dpi (≈2400×1350 px), larger
axis labels, bold quadrant ticks, WHO ≥ 2 h benchmark line.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_vidigal_aspect_avg_winter_v2.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
from fig_style import apply_style  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from src.morphometry.aspect import aspect_quadrant  # noqa: E402
from src.viz import presentation_style as ps  # noqa: E402

PAPER_OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
PRESENTATION_OUT_DIR = PROJECT_ROOT / "outputs" / "vidigal" / "presentation_figures"
OUT_FILENAME = "fig_vidigal_aspect_avg_winter_v2.png"

QUADRANTS = ("N", "E", "S", "W")
ACCENT_S = "#c0392b"
BAR_DEFAULT = "#475569"  # slate (existing palette)
WHO_THRESHOLD = 2.0


def load_points() -> gpd.GeoDataFrame:
    solar_path = (
        PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "svf"
        / "svf_streets_solar.gpkg"
    )
    grid_path = (
        PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "grid"
        / "grid_metrics.gpkg"
    )
    solar = gpd.read_file(solar_path)
    grid = gpd.read_file(grid_path)
    if solar.crs != grid.crs:
        solar = solar.to_crs(grid.crs)
    keep = grid[["geometry", "aspect_deg", "slope_deg"]]
    pts = gpd.sjoin(solar, keep, how="left", predicate="within").drop(columns="index_right")
    pts = pts.dropna(subset=["aspect_deg", "solar_hours_winter"])
    pts["quadrant"] = aspect_quadrant(pts["aspect_deg"].values)
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=("paper", "presentation"), default="paper")
    args = parser.parse_args()

    apply_style()
    ps.apply(args.preset)
    pts = load_points()
    print(f"  Loaded {len(pts):,} street points")

    means, p25, p75 = [], [], []
    for q in QUADRANTS:
        vals = pts.loc[pts["quadrant"] == q, "solar_hours_winter"].values
        means.append(float(np.mean(vals)))
        p25.append(float(np.percentile(vals, 25)))
        p75.append(float(np.percentile(vals, 75)))
        print(f"    {q}: n={len(vals):,}  mean={means[-1]:.2f}  P25={p25[-1]:.2f}  P75={p75[-1]:.2f}")

    means = np.array(means)
    p25 = np.array(p25)
    p75 = np.array(p75)
    err_lo = means - p25
    err_hi = p75 - means

    # 16:9, 8 in × 4.5 in @ 300 dpi → 2400 × 1350 px
    fig_w, fig_h = 8.0, 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = [ACCENT_S if q == "S" else BAR_DEFAULT for q in QUADRANTS]
    x = np.arange(len(QUADRANTS))
    ax.bar(
        x, means, width=0.62, color=colors,
        edgecolor="black", linewidth=1.2, zorder=3,
    )
    ax.errorbar(
        x, means, yerr=[err_lo, err_hi],
        fmt="none", ecolor="#aaaaaa", elinewidth=1.6, capsize=8, capthick=1.4,
        zorder=4,
    )

    y_max = float(np.max(p75)) * 1.20
    for xi, m in zip(x, means):
        ax.text(
            xi, m + y_max * 0.022, f"{m:.2f}",
            ha="center", va="bottom",
            fontsize=15, fontweight="bold", color="#111111", zorder=6,
        )

    # WHO ≥ 2 h benchmark — label in the S/W gap above the line
    ax.axhline(WHO_THRESHOLD, color="#888888", linestyle="--", linewidth=1.4, zorder=2)
    ax.text(
        2.5, WHO_THRESHOLD + y_max * 0.012,
        "WHO ≥ 2 h",
        ha="center", va="bottom", fontsize=13,
        color="#555555", fontstyle="italic", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(list(QUADRANTS), fontsize=20, fontweight="bold")
    ax.set_ylabel("mean winter direct-sun (h)", fontsize=16)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis="y", labelsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(True, axis="y", color="#dddddd", lw=0.6, zorder=0)

    plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.13)

    out_dir = PRESENTATION_OUT_DIR if args.preset == "presentation" else PAPER_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / OUT_FILENAME
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    size_kb = out_png.stat().st_size // 1024
    print(f"  Saved {out_png} ({size_kb} KB, {fig_w:.1f}x{fig_h:.2f} in @ 300 dpi)")


if __name__ == "__main__":
    main()
