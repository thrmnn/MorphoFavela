#!/usr/bin/env python3
"""BRISA slide deck — fig_vidigal_aspect_avg_winter.png.

Vidigal-only. x = aspect quadrant (N · E · S · W), y = mean winter-solstice
direct-sun hours, averaged across all street points in each quadrant (no
slope binning). Simple bar chart with the S-facing bar in red to draw the
eye to the low-sun quadrant. P25/P75 shown as light grey error whiskers.

Same data source as fig_vidigal_aspect_slope_curves.py.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_vidigal_aspect_avg_winter.png
"""
from __future__ import annotations

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

OUT_DIR = Path("/home/theo/brisa_paper/artifacts/slides/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUADRANTS = ("N", "E", "S", "W")
ACCENT_S = "#c0392b"  # red on south-facing
BAR_DEFAULT = "#333333"


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
    apply_style()
    pts = load_points()
    print(f"  Loaded {len(pts):,} street points")

    means, p25, p75, counts = [], [], [], []
    for q in QUADRANTS:
        vals = pts.loc[pts["quadrant"] == q, "solar_hours_winter"].values
        means.append(float(np.mean(vals)))
        p25.append(float(np.percentile(vals, 25)))
        p75.append(float(np.percentile(vals, 75)))
        counts.append(len(vals))
        print(f"    {q}: n={len(vals):,}  mean={means[-1]:.2f}  P25={p25[-1]:.2f}  P75={p75[-1]:.2f}")

    means = np.array(means)
    p25 = np.array(p25)
    p75 = np.array(p75)
    err_lo = means - p25
    err_hi = p75 - means

    fig_w = 18.0
    fig_h = fig_w / 1.4  # ≈ 12.86 in
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = [ACCENT_S if q == "S" else BAR_DEFAULT for q in QUADRANTS]
    x = np.arange(len(QUADRANTS))
    bars = ax.bar(
        x, means, width=0.62, color=colors,
        edgecolor="black", linewidth=1.4, zorder=3,
    )
    ax.errorbar(
        x, means, yerr=[err_lo, err_hi],
        fmt="none", ecolor="#aaaaaa", elinewidth=2.0, capsize=10, capthick=1.8,
        zorder=4,
    )

    y_max = float(np.max(p75)) * 1.18
    for xi, m in zip(x, means):
        ax.text(
            xi, m + y_max * 0.025, f"{m:.2f}",
            ha="center", va="bottom",
            fontsize=22, fontweight="bold", color="#111111", zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(QUADRANTS), fontsize=24)
    ax.set_ylabel("mean winter direct-sun (h)", fontsize=22)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis="y", labelsize=20)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(True, axis="y", color="#dddddd", lw=0.8, zorder=0)

    plt.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.12)

    out = OUT_DIR / "fig_vidigal_aspect_avg_winter.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB, {fig_w:.1f}x{fig_h:.2f} in @ 200 dpi)")


if __name__ == "__main__":
    main()
