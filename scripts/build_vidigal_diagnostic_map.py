#!/usr/bin/env python3
"""Build the Vidigal partial-constraint diagnostic map for the BRISA paper.

Four-state map at 10 m grid resolution:
    - adequate                (both thresholds pass)
    - sunlight constraint     (winter direct sun < 2 h)
    - ventilation constraint  (lambda_f > 0.35, skimming-flow proxy)
    - compound constraint     (both)

Sunlight signal is real (ray-cast winter solstice solar hours per street
observation point, median-aggregated to grid cells). Ventilation signal is
a geometric proxy from frontal-area density — explicit pre-CFD diagnostic,
to be replaced by the CFD-derived ACH when the campaign completes.

Output:
    outputs/vidigal/paper_figures/fig_vidigal_diagnostic_map.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.viz import presentation_style as ps

SITE = "vidigal"

GRID_PATH = PROJECT_ROOT / "outputs" / SITE / "morphometrics" / "grid" / "grid_metrics.gpkg"
SOLAR_PATH = PROJECT_ROOT / "outputs" / SITE / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
BLDG_PATH = PROJECT_ROOT / "data" / SITE / "buildings_extended_300m.gpkg"
PAPER_OUT_DIR = PROJECT_ROOT / "outputs" / SITE / "paper_figures"
PRESENTATION_OUT_DIR = PROJECT_ROOT / "outputs" / SITE / "presentation_figures"
OUT_FILENAME = "fig_vidigal_diagnostic_map.png"

THRESHOLD_SUN_HRS = 2.0      # WHO winter direct-sun floor.
THRESHOLD_LAMBDA_F = 0.35    # Skimming-flow regime (Grimmond & Oke 1999).

STATE_ADEQUATE = 0
STATE_SUN_ONLY = 1
STATE_VENT_ONLY = 2
STATE_COMPOUND = 3
STATE_NODATA = 4

STATE_COLORS = ["#FFFFFF", "#D9D9D9", "#7F7F7F", "#E76F51", "#F4F0E8"]
STATE_LABELS = [
    "adequate (both thresholds pass)",
    f"sunlight constraint (winter sun < {THRESHOLD_SUN_HRS:.0f} h)",
    f"ventilation constraint (λf > {THRESHOLD_LAMBDA_F:.2f})",
    "compound constraint (both)",
    "no data",
]


def aggregate_solar_to_cells(
    grid: gpd.GeoDataFrame, max_radius: float = 25.0, k: int = 3
) -> gpd.GeoDataFrame:
    """Aggregate winter solar hours to grid cells.

    First tries within-cell median; falls back to nearest-k street
    observations within max_radius for cells without an internal sample.
    """
    sol = gpd.read_file(SOLAR_PATH)[["solar_hours_winter", "geometry"]]
    j = gpd.sjoin(sol, grid[["zone_id", "geometry"]], how="inner", predicate="within")
    primary = j.groupby("zone_id")["solar_hours_winter"].median().reset_index()
    grid = grid.merge(primary, on="zone_id", how="left")

    missing = grid["solar_hours_winter"].isna()
    if missing.any():
        sol_pts = np.array([(g.x, g.y) for g in sol.geometry])
        sol_vals = sol["solar_hours_winter"].to_numpy()
        tree = cKDTree(sol_pts)
        cents = grid.loc[missing, ["centroid_x", "centroid_y"]].to_numpy()
        dists, idxs = tree.query(cents, k=k, distance_upper_bound=max_radius)
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(dists)
        idxs = np.where(valid, idxs, 0)
        vals = sol_vals[idxs]
        vals = np.where(valid, vals, np.nan)
        with np.errstate(invalid="ignore"):
            cell_vals = np.nanmedian(vals, axis=1)
        grid.loc[missing, "solar_hours_winter"] = cell_vals
    return grid


def classify(grid: gpd.GeoDataFrame) -> np.ndarray:
    """Built cells only (lambda_p > 0); empty-terrain cells stay no-data."""
    sun = grid["solar_hours_winter"]
    vent = grid["lambda_f_mean"]
    state = np.full(len(grid), STATE_NODATA, dtype=int)
    built = (grid["lambda_p"].fillna(0) > 0.01) | (grid["building_count"] > 0)
    sun_known = sun.notna()
    vent_known = vent.notna()
    both = built & sun_known & vent_known
    sun_fail = both & (sun < THRESHOLD_SUN_HRS)
    vent_fail = both & (vent > THRESHOLD_LAMBDA_F)
    state[both & ~sun_fail & ~vent_fail] = STATE_ADEQUATE
    state[both & sun_fail & ~vent_fail] = STATE_SUN_ONLY
    state[both & ~sun_fail & vent_fail] = STATE_VENT_ONLY
    state[both & sun_fail & vent_fail] = STATE_COMPOUND
    return state


def render(grid: gpd.GeoDataFrame, state: np.ndarray, preset: str = "paper") -> None:
    fig, ax = plt.subplots(figsize=(12, 6.0), facecolor="white")
    ax.set_facecolor("white")

    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)

    grid_to_plot = grid.copy()
    grid_to_plot["state"] = state
    classified = grid_to_plot[grid_to_plot["state"] != STATE_NODATA]
    classified.plot(
        ax=ax, column="state", cmap=cmap, norm=norm,
        edgecolor="#FFFFFF", linewidth=0.12,
    )

    if BLDG_PATH.exists():
        bldg = gpd.read_file(BLDG_PATH)
        if bldg.crs != grid.crs:
            bldg = bldg.to_crs(grid.crs)
        bldg.boundary.plot(ax=ax, color="#000000", linewidth=0.25, alpha=0.55)

    bbox = classified.total_bounds
    pad = 25.0
    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)
    ax.set_aspect("equal")
    if preset == "presentation":
        ps.apply_to_map_axes(ax)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    counts = {s: int((state == s).sum()) for s in range(5)}
    total_known = sum(counts[s] for s in (STATE_ADEQUATE, STATE_SUN_ONLY,
                                          STATE_VENT_ONLY, STATE_COMPOUND))
    legend_handles = []
    for s in (STATE_ADEQUATE, STATE_SUN_ONLY, STATE_VENT_ONLY, STATE_COMPOUND):
        share = counts[s] / total_known if total_known else 0.0
        label = f"{STATE_LABELS[s]}  —  {share*100:.0f}%"
        legend_handles.append(mpatches.Patch(
            facecolor=STATE_COLORS[s], edgecolor="#555555", linewidth=0.6,
            label=label,
        ))
    legend_fs = 13 if preset == "presentation" else 10
    leg = ax.legend(
        handles=legend_handles, loc="lower left", frameon=False,
        bbox_to_anchor=(0.0, -0.12), fontsize=legend_fs, handlelength=1.6,
        handleheight=1.0, borderpad=0.4, labelspacing=0.6, ncol=2,
        columnspacing=2.0,
    )
    for txt in leg.get_texts():
        txt.set_color("#222222")

    if preset != "presentation":
        ax.text(
            0.01, 1.02,
            "VIDIGAL  ·  diagnostic map (pre-CFD; geometric λf proxy for ventilation)",
            transform=ax.transAxes, fontsize=9.5, color="#666666",
            ha="left", va="bottom",
        )

    if preset == "presentation":
        ps.add_scale_bar(ax, 100.0, loc="lower right")
    else:
        bar_y = 0.02
        bar_x0 = 0.78
        scale_m = 100
        bbox_data = grid.total_bounds
        span = bbox_data[2] - bbox_data[0]
        bar_w = scale_m / span * (1 - 0.02)
        ax.add_patch(mpatches.Rectangle(
            (bar_x0, bar_y), bar_w, 0.008,
            transform=ax.transAxes, color="#222222",
        ))
        ax.text(
            bar_x0 + bar_w / 2, bar_y + 0.018,
            f"{scale_m} m", transform=ax.transAxes,
            fontsize=8.5, color="#444444", ha="center", va="bottom",
        )

    out_dir = PRESENTATION_OUT_DIR if preset == "presentation" else PAPER_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / OUT_FILENAME
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nwrote {out_png}")
    print(f"n cells: {len(grid)}")
    print(f"  adequate:        {counts[STATE_ADEQUATE]:5d}  ({counts[STATE_ADEQUATE]/total_known*100:5.1f}%)")
    print(f"  sunlight constraint:     {counts[STATE_SUN_ONLY]:5d}  ({counts[STATE_SUN_ONLY]/total_known*100:5.1f}%)")
    print(f"  ventilation constraint:  {counts[STATE_VENT_ONLY]:5d}  ({counts[STATE_VENT_ONLY]/total_known*100:5.1f}%)")
    print(f"  compound constraint:     {counts[STATE_COMPOUND]:5d}  ({counts[STATE_COMPOUND]/total_known*100:5.1f}%)")
    print(f"  no data:         {counts[STATE_NODATA]:5d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=("paper", "presentation"), default="paper")
    args = parser.parse_args()

    ps.apply(args.preset)
    grid = gpd.read_file(GRID_PATH)
    grid = aggregate_solar_to_cells(grid)
    state = classify(grid)
    render(grid, state, preset=args.preset)


if __name__ == "__main__":
    main()
