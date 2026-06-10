#!/usr/bin/env python3
"""BRISA slide deck — fig_facade_vidigal_pilot.png.

REAL Vidigal facade-level winter-solstice (2026-06-21) ray-cast results.

Data source:
  outputs/vidigal_tls/solar/facade/facade_solar_20260621.gpkg
    — 92,083 facade sample points, each with facade_azimuth, height_above_ground,
      facade_sunlit_hours, who_compliant, building_id
  data/vidigal_tls/raw/vidigal_buildings.shp  — 436 footprints

Layout (16:9):
  LEFT  — plan view of Vidigal TLS building footprints, coloured by per-building
          mean facade direct-sun hours on winter solstice
  RIGHT — four histograms of direct-sun-hours, one per facade quadrant
          (N · E · S · W), with WHO ≥2h benchmark line and the
          % of facade samples below WHO annotated.

Headline finding:
  South-facing facades: 97.7% of samples below WHO 2h benchmark on winter
  solstice. N: 51.4%. E: 44.0%. W: 83.8%.

Output: /home/theo/brisa_paper/artifacts/slides/assets/fig_facade_vidigal_pilot.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
sys.path.insert(0, str(PAPER_FIG_DIR))
from fig_style import apply_style  # noqa: E402

OUT = Path("/home/theo/brisa_paper/artifacts/slides/assets/fig_facade_vidigal_pilot.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

FACADE_GPKG = PROJECT_ROOT / "outputs" / "vidigal_tls" / "solar" / "facade" / "facade_solar_20260621.gpkg"
BUILDINGS_SHP = PROJECT_ROOT / "data" / "vidigal_tls" / "raw" / "vidigal_buildings.shp"

WHO_HOURS = 2.0

# Quadrant colours: keep S in the warning red (it is the failing quadrant).
QUAD_COL = {
    "N": "#4a7a98",  # cool blue — well-lit
    "E": "#7a9c5a",  # green — well-lit
    "S": "#c0392b",  # red — failing
    "W": "#b8893f",  # ochre — intermediate
}


def assign_quadrant(az: np.ndarray) -> np.ndarray:
    """Bin facade azimuth (deg, 0=N, 90=E, 180=S, 270=W) into 4 cardinal quadrants."""
    a = az % 360
    out = np.empty(a.shape, dtype="U1")
    out[(a < 45) | (a >= 315)] = "N"
    out[(a >= 45) & (a < 135)] = "E"
    out[(a >= 135) & (a < 225)] = "S"
    out[(a >= 225) & (a < 315)] = "W"
    return out


def main():
    apply_style()

    # ── Load data ────────────────────────────────────────────────────────────
    facade = gpd.read_file(FACADE_GPKG)
    print(f"Loaded {len(facade):,} facade points")
    facade["quad"] = assign_quadrant(facade["facade_azimuth"].values)

    buildings = gpd.read_file(BUILDINGS_SHP).to_crs(facade.crs)
    bldg_mean = facade.groupby("building_id")["facade_sunlit_hours"].mean().rename("mean_h")
    buildings = buildings.merge(
        bldg_mean, left_index=True, right_index=True, how="left"
    )
    # Some building_ids in the facade df may not match positional index; fall back
    # to ID-based merge if mean_h is mostly NaN.
    if buildings["mean_h"].notna().sum() < 50:
        # Use building_id column if present
        if "building_id" in buildings.columns:
            buildings = buildings.drop(columns="mean_h").merge(
                bldg_mean, left_on="building_id", right_index=True, how="left"
            )
    print(f"Buildings with facade mean: {buildings['mean_h'].notna().sum()} / {len(buildings)}")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig_w, fig_h = 16.0, 9.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        4, 2,
        width_ratios=[1.05, 1.0],
        height_ratios=[1, 1, 1, 1],
        wspace=0.18, hspace=0.55,
        left=0.045, right=0.985, top=0.93, bottom=0.085,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_hists = [fig.add_subplot(gs[i, 1]) for i in range(4)]

    # ── LEFT PANEL — plan map ────────────────────────────────────────────────
    cmap_name = "facade_sun"
    facade_cmap = LinearSegmentedColormap.from_list(
        cmap_name,
        ["#2a2a35", "#7a3a3a", "#c0723a", "#e8b757", "#f5e58a"],
    )
    vmin, vmax = 0.0, 6.0  # data range cap
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Background: light grey footprints for ones missing data
    missing = buildings[buildings["mean_h"].isna()]
    if len(missing):
        missing.plot(ax=ax_map, facecolor="#e5e2dc", edgecolor="#bbb", linewidth=0.2, zorder=2)

    with_data = buildings[buildings["mean_h"].notna()]
    with_data.plot(
        ax=ax_map,
        column="mean_h", cmap=facade_cmap, vmin=vmin, vmax=vmax,
        edgecolor="#222", linewidth=0.3, zorder=3,
    )

    # Equal aspect, no spines
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for s in ax_map.spines.values():
        s.set_visible(False)

    # North arrow
    xlim = ax_map.get_xlim()
    ylim = ax_map.get_ylim()
    ex = xlim[1] - xlim[0]
    ey = ylim[1] - ylim[0]
    ax_map.annotate(
        "", xy=(xlim[1] - ex * 0.06, ylim[1] - ey * 0.06),
        xytext=(xlim[1] - ex * 0.06, ylim[1] - ey * 0.13),
        arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0),
    )
    ax_map.text(xlim[1] - ex * 0.06, ylim[1] - ey * 0.05, "N",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Scale bar — 50 m
    scale_m = 50.0
    sx0 = xlim[0] + ex * 0.06
    sy0 = ylim[0] + ey * 0.05
    ax_map.plot([sx0, sx0 + scale_m], [sy0, sy0], color="k", lw=2.2, solid_capstyle="butt")
    ax_map.text(sx0 + scale_m / 2, sy0 + ey * 0.018, f"{int(scale_m)} m",
                ha="center", va="bottom", fontsize=10)

    # Map subtitle
    ax_map.set_title(
        "Vidigal TLS (436 buildings)\nmean facade direct-sun hours, winter solstice",
        fontsize=14, loc="left", color="#222", pad=8,
    )

    # Colorbar inside map axis (bottom)
    cax = fig.add_axes([0.065, 0.06, 0.36, 0.018])
    sm = ScalarMappable(norm=norm, cmap=facade_cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("direct-sun hours (winter solstice)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_linewidth(0.4)

    # ── RIGHT PANEL — 4 histograms ───────────────────────────────────────────
    bins = np.arange(0, 10.5, 0.5)
    quad_order = ["N", "E", "S", "W"]
    # Compute per-quadrant counts for shared ylim. The 0-bin dominates everywhere
    # (lots of fully-shaded samples), so we cap ylim at the max of the *non-zero*
    # bins, then clip the 0-bin visually and tag it with its true % share.
    nonzero_max = 0
    zero_shares = {}
    quad_data = {}
    for q in quad_order:
        h = facade.loc[facade["quad"] == q, "facade_sunlit_hours"].values
        quad_data[q] = h
        c, _ = np.histogram(h, bins=bins)
        nonzero_max = max(nonzero_max, c[1:].max())  # skip the 0-h bin
        zero_shares[q] = (h == 0).mean() * 100

    ylim_top = nonzero_max * 1.45

    for ax, q in zip(ax_hists, quad_order):
        h = quad_data[q]
        n = len(h)
        below = (h < WHO_HOURS).mean() * 100
        mean_h = h.mean()
        col = QUAD_COL[q]

        counts, _ = np.histogram(h, bins=bins)
        centers = (bins[:-1] + bins[1:]) / 2
        widths = np.diff(bins)
        # Clip the zero-bin to ylim_top and tag it; draw remaining bars normally.
        zero_count = counts[0]
        clipped = counts.copy()
        clipped[0] = min(zero_count, int(ylim_top * 0.95))
        ax.bar(centers, clipped, width=widths * 0.92,
               color=col, edgecolor="white", linewidth=0.6, zorder=3, align="center")

        # Annotate clipped 0-bin if it was clipped — place to the right of the
        # 0-bin so it doesn't get cut off, dark text on background.
        if zero_count > ylim_top * 0.95:
            ax.text(
                centers[0] + widths[0] * 0.7, ylim_top * 0.92,
                f"{zero_shares[q]:.0f}% at 0 h ↑",
                ha="left", va="top", fontsize=10, color=col, fontweight="bold",
                zorder=5,
            )

        ax.axvline(WHO_HOURS, color="#222", lw=1.2, ls="--", zorder=4)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, ylim_top)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="x", labelsize=10, length=3)

        if q != "W":
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("direct-sun hours (winter solstice)", fontsize=11)

        # Quadrant tag, big — left side
        ax.text(
            -0.04, 0.5, q,
            transform=ax.transAxes, ha="right", va="center",
            fontsize=32, fontweight="bold", color=col,
        )

        # Per-quadrant readout, right side
        readout = f"n = {n:,}   mean = {mean_h:.2f} h   below WHO 2h = {below:.0f}%"
        ax.text(
            0.99, 0.94, readout,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11.5, color="#222",
        )

        # WHO label only on top histogram to avoid clutter
        if q == "N":
            ax.text(
                WHO_HOURS + 0.15, ylim_top * 0.55,
                "WHO 2h", fontsize=10, color="#222", style="italic",
                ha="left", va="top",
            )

    # Right-panel section heading
    fig.text(
        0.745, 0.96,
        "facade direct-sun distribution by orientation quadrant",
        ha="center", va="top", fontsize=13, color="#222",
    )

    # ── Footer / data provenance ─────────────────────────────────────────────
    fig.text(
        0.5, 0.015,
        "Vidigal · facade ray-cast, 30-min interval, 0.2 m self-intersection cull · n = 92,083 facade points · WHO 2h winter-solstice benchmark",
        ha="center", va="bottom", fontsize=8.5, color="#666", style="italic",
    )

    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
