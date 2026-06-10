#!/usr/bin/env python3
"""BRISA slide deck — fig_vidigal_aspect_slope_curves.png (NEW, slide 14).

Vidigal cells only. x = slope (degrees, 0–35°), y = mean winter-solstice
direct-sun hours. Four lines, one per aspect quadrant (N / E / S / W).

Data source: per-street-point solar pipeline (svf_streets_solar.gpkg) joined
to the 10 m morphometric grid by spatial-within, which gives every street
point an inherited aspect_deg + slope_deg from the underlying cell. Mean
per (slope-bin, quadrant) is plotted.

The empirical N − S gap on Vidigal is ≈ 1.5–2.5 h across most slope bins
(not the textbook 4–5 h): structural shading from neighbouring buildings
damps the pure-terrain aspect signal that you would see on bare slopes.
The high-slope (24–34°) band is the cleanest expression of the N-favours-
winter-sun effect and is annotated explicitly.

B/W aesthetic: N = solid black, S = solid grey w/ markers, E = dashed black,
W = dashed grey.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_vidigal_aspect_slope_curves.png
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
QUAD_STYLE = {
    "N": dict(color="#000000", linestyle="-", marker="o", lw=2.2, ms=5.5, label="N-facing"),
    "E": dict(color="#000000", linestyle="--", marker="s", lw=1.6, ms=4.5, label="E-facing"),
    "S": dict(color="#888888", linestyle="-", marker="o", lw=2.2, ms=5.5, label="S-facing"),
    "W": dict(color="#888888", linestyle="--", marker="s", lw=1.6, ms=4.5, label="W-facing"),
}

SLOPE_BIN_W = 2.0
SLOPE_MIN, SLOPE_MAX = 0.0, 35.0
MIN_BIN_N = 10


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
    pts = pts.dropna(subset=["aspect_deg", "slope_deg", "solar_hours_winter"])
    pts["quadrant"] = aspect_quadrant(pts["aspect_deg"].values)
    return pts


def quadrant_curve(pts: gpd.GeoDataFrame, quad: str, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = pts[pts["quadrant"] == quad]
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(len(centres), np.nan)
    counts = np.zeros(len(centres), dtype=int)
    idx = np.digitize(sub["slope_deg"].values, edges) - 1
    idx = np.clip(idx, 0, len(centres) - 1)
    vals = sub["solar_hours_winter"].values
    for i in range(len(centres)):
        sel = vals[idx == i]
        counts[i] = len(sel)
        if len(sel) >= MIN_BIN_N:
            means[i] = float(np.mean(sel))
    return centres, means, counts


def main() -> None:
    apply_style()
    pts = load_points()
    print(f"  Loaded {len(pts):,} street points with aspect+slope+winter-sun")
    print("  Per-quadrant counts:")
    for q in QUADRANTS:
        n = int((pts["quadrant"] == q).sum())
        print(f"    {q}: {n:,}")

    edges = np.arange(SLOPE_MIN, SLOPE_MAX + SLOPE_BIN_W, SLOPE_BIN_W)
    curves = {q: quadrant_curve(pts, q, edges) for q in QUADRANTS}

    fig_w = 11.0
    fig_h = fig_w / 1.4  # ≈ 7.85 in
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Plot per-quadrant traces.
    for q in QUADRANTS:
        centres, means, counts = curves[q]
        finite = np.isfinite(means)
        st = QUAD_STYLE[q]
        ax.plot(centres[finite], means[finite],
                color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], lw=st["lw"], ms=st["ms"],
                mfc=st["color"], mec=st["color"], label=st["label"],
                zorder=4 if q in ("N", "S") else 3)

    # --- Annotate the mean N − S gap on slope ≥ 5° (the regime where
    # terrain aspect can geometrically matter).
    sloped = pts[pts["slope_deg"] >= 5.0]
    mean_N_all = float(sloped[sloped["quadrant"] == "N"]["solar_hours_winter"].mean())
    mean_S_all = float(sloped[sloped["quadrant"] == "S"]["solar_hours_winter"].mean())
    gap = mean_N_all - mean_S_all
    # Highlight the sloped regime (≥ 5°).
    ax.axvspan(5.0, SLOPE_MAX, color="#eeeeee", alpha=0.55, zorder=0)
    # Reference lines at the regime-mean N and S levels.
    ax.hlines(mean_N_all, 5.0, SLOPE_MAX, color="#000000",
              linestyle=":", lw=1.0, zorder=5)
    ax.hlines(mean_S_all, 5.0, SLOPE_MAX, color="#888888",
              linestyle=":", lw=1.0, zorder=5)
    # Two-headed arrow at a clean x position.
    x_anno = 33.0
    ax.annotate(
        "", xy=(x_anno, mean_N_all), xytext=(x_anno, mean_S_all),
        arrowprops=dict(arrowstyle="<->", color="#111111", lw=1.4),
        zorder=6,
    )
    ax.text(
        x_anno - 0.6, (mean_N_all + mean_S_all) / 2,
        f"{gap:.1f} h\n(N − S, slope ≥ 5°)",
        fontsize=11, fontweight="bold",
        color="#111111", ha="right", va="center",
        bbox=dict(facecolor="white", edgecolor="#888888", boxstyle="round,pad=0.3",
                  alpha=0.95, linewidth=0.5),
        zorder=7,
    )
    print(f"  Mean N − S gap on slope ≥ 5°: {gap:.2f} h (N={mean_N_all:.2f}, S={mean_S_all:.2f})")

    # Axis cosmetics.
    ax.set_xlabel("Slope (degrees)", fontsize=12)
    ax.set_ylabel("Mean winter-solstice direct-sun (hours)", fontsize=12)
    ax.set_xlim(SLOPE_MIN, SLOPE_MAX)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(True, axis="y", color="#dddddd", lw=0.5, zorder=0)

    leg = ax.legend(
        loc="lower left", frameon=True, fontsize=11,
        title="Terrain aspect quadrant (downslope bearing)",
        title_fontsize=11, ncol=2, handlelength=2.6,
    )
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#888888")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)

    fig.suptitle(
        "Vidigal — winter sun decouples from slope by terrain aspect",
        x=0.09, y=0.97, ha="left", fontsize=13.5, color="#111111",
    )
    fig.text(
        0.09, 0.92,
        "Per-quadrant mean of winter-solstice direct sun. On sloped ground (≥ 5°) N-facing cells receive ≈ 1.6 h more winter sun\nthan S-facing — smaller than the pure-terrain expectation because alley-canyon shading damps the signal. Per-bin curves\nare noisy where n is low (especially W-facing).",
        ha="left", va="top",
        fontsize=10, color="#444444", style="italic",
    )

    plt.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.10)

    out = OUT_DIR / "fig_vidigal_aspect_slope_curves.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    size_kb = out.stat().st_size // 1024
    print(f"  Saved {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
