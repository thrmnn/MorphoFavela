#!/usr/bin/env python3
"""Build the partial-constraint diagnostic map for any BRISA site.

Four-state map at 10 m grid resolution:
    - adequate                (both thresholds pass)
    - sunlight constraint     (winter direct sun < 2 h)
    - ventilation constraint  (lambda_f > 0.35, skimming-flow proxy)
    - compound constraint     (both)

Sunlight signal: ray-cast winter-solstice solar hours per street observation
point, median-aggregated to grid cells with nearest-k fallback. Ventilation
signal: geometric proxy from frontal-area density; literature anchor
Grimmond & Oke (1999). Pre-CFD diagnostic, to be replaced by CFD-derived
ACH when the OpenFOAM campaign completes.

Usage:
    python scripts/build_diagnostic_map.py --site vidigal
    python scripts/build_diagnostic_map.py --site rocinha
    python scripts/build_diagnostic_map.py --site complexo_do_alemao
    python scripts/build_diagnostic_map.py --site maré
    python scripts/build_diagnostic_map.py --site riodaspedras
    python scripts/build_diagnostic_map.py --all      # run every site

Output per site:
    outputs/<site>/paper_figures/fig_<site>_diagnostic_map.png
    outputs/<site>/paper_figures/diagnostic_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial import cKDTree

# Hardcoded, not Path(__file__).resolve().parents[1]: this script also runs
# from a git worktree that has no data/outputs/ of its own (same rationale
# as scripts/build_site_dashboard.py, scripts/build_html_dashboard.py) — it
# always reads/writes the one main checkout's data/outputs regardless of
# which checkout's copy of this file is running. sys.path, unlike
# PROJECT_ROOT, uses the RUNNING checkout's own root (whichever it is) so
# `import src...` resolves that checkout's own src/ tree (e.g. a module
# added on a worktree branch, not yet merged to the main checkout).
PROJECT_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "maré", "riodaspedras"]
SITE_DISPLAY = {
    "vidigal": "VIDIGAL",
    "rocinha": "ROCINHA",
    "complexo_do_alemao": "COMPLEXO DO ALEMÃO",
    "maré": "MARÉ",
    "riodaspedras": "RIO DAS PEDRAS",
}

THRESHOLD_SUN_HRS = 2.0  # WHO winter direct-sun floor.
THRESHOLD_LAMBDA_F = 0.35  # Skimming-flow regime (Grimmond & Oke 1999).

STATE_ADEQUATE = 0
STATE_SUN_ONLY = 1
STATE_VENT_ONLY = 2
STATE_COMPOUND = 3
STATE_NODATA = 4

STATE_COLORS = ["#FFFFFF", "#D9D9D9", "#7F7F7F", "#111111", "#F4F0E8"]
STATE_LABELS = [
    "adequate (both thresholds pass)",
    f"sunlight constraint (winter sun < {THRESHOLD_SUN_HRS:.0f} h)",
    f"ventilation constraint (λf > {THRESHOLD_LAMBDA_F:.2f})",
    "compound constraint (both)",
    "no data",
]

# JSON / CSV / GPKG attribute keys for the four diagnostic states.
STATE_KEYS = {
    STATE_ADEQUATE: "adequate",
    STATE_SUN_ONLY: "sunlight_constraint",
    STATE_VENT_ONLY: "ventilation_constraint",
    STATE_COMPOUND: "compound_constraint",
}


def site_paths(site: str, study_area: bool = False) -> dict:
    """`study_area=True` (Maré only) writes to `*_study_area` suffixed
    outputs instead of the default whole-data-extent ones, so the
    Maré-brief/site-sheet study-area recompute never overwrites the
    whole-bairro diagnostic map/stats other consumers read (technical
    report, manuscript figures, project hub)."""
    suffix = "_study_area" if study_area else ""
    return {
        "grid": PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg",
        "solar": PROJECT_ROOT
        / "outputs"
        / site
        / "morphometrics"
        / "svf"
        / "svf_streets_solar.gpkg",
        "bldg": PROJECT_ROOT / "data" / site / "buildings_extended_300m.gpkg",
        "out_dir": PROJECT_ROOT / "outputs" / site / "paper_figures",
        "out_png": PROJECT_ROOT
        / "outputs"
        / site
        / "paper_figures"
        / f"fig_{site}_diagnostic_map{suffix}.png",
        "out_stats": PROJECT_ROOT / "outputs" / site / "paper_figures" / f"diagnostic_stats{suffix}.json",
    }


def aggregate_solar_to_cells(
    grid: gpd.GeoDataFrame,
    solar_path: Path,
    max_radius: float = 25.0,
    k: int = 3,
) -> gpd.GeoDataFrame:
    """Aggregate winter solar hours to grid cells.

    First tries within-cell median; falls back to nearest-k street
    observations within max_radius for cells without an internal sample.
    """
    sol = gpd.read_file(solar_path)[["solar_hours_winter", "geometry"]]
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


def render(site: str, grid: gpd.GeoDataFrame, state: np.ndarray, paths: dict,
           title_suffix: str = "") -> dict:
    fig, ax = plt.subplots(figsize=(12, 6.0), facecolor="white")
    ax.set_facecolor("white")

    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)

    grid_to_plot = grid.copy()
    grid_to_plot["state"] = state
    classified = grid_to_plot[grid_to_plot["state"] != STATE_NODATA]
    classified.plot(
        ax=ax,
        column="state",
        cmap=cmap,
        norm=norm,
        edgecolor="#FFFFFF",
        linewidth=0.12,
    )

    if paths["bldg"].exists():
        bldg = gpd.read_file(paths["bldg"])
        if bldg.crs != grid.crs:
            bldg = bldg.to_crs(grid.crs)
        bldg.boundary.plot(ax=ax, color="#000000", linewidth=0.25, alpha=0.55)

    bbox = classified.total_bounds
    pad = 25.0
    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    counts = {s: int((state == s).sum()) for s in range(5)}
    total_known = sum(
        counts[s] for s in (STATE_ADEQUATE, STATE_SUN_ONLY, STATE_VENT_ONLY, STATE_COMPOUND)
    )
    legend_handles = []
    for s in (STATE_ADEQUATE, STATE_SUN_ONLY, STATE_VENT_ONLY, STATE_COMPOUND):
        share = counts[s] / total_known if total_known else 0.0
        label = f"{STATE_LABELS[s]}  —  {share * 100:.0f}%"
        legend_handles.append(
            mpatches.Patch(
                facecolor=STATE_COLORS[s],
                edgecolor="#555555",
                linewidth=0.6,
                label=label,
            )
        )
    leg = ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        bbox_to_anchor=(0.0, -0.12),
        fontsize=10,
        handlelength=1.6,
        handleheight=1.0,
        borderpad=0.4,
        labelspacing=0.6,
        ncol=2,
        columnspacing=2.0,
    )
    for txt in leg.get_texts():
        txt.set_color("#222222")

    ax.text(
        0.01,
        1.02,
        f"{SITE_DISPLAY[site]}{title_suffix}  ·  diagnostic map (pre-CFD; geometric λf proxy for ventilation)",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#666666",
        ha="left",
        va="bottom",
    )

    bar_y = 0.02
    bar_x0 = 0.78
    scale_m = 100
    bbox_full = grid.total_bounds
    span = bbox_full[2] - bbox_full[0]
    bar_w = scale_m / span * (1 - 0.02)
    ax.add_patch(
        mpatches.Rectangle(
            (bar_x0, bar_y),
            bar_w,
            0.008,
            transform=ax.transAxes,
            color="#222222",
        )
    )
    ax.text(
        bar_x0 + bar_w / 2,
        bar_y + 0.018,
        f"{scale_m} m",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
        ha="center",
        va="bottom",
    )

    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    plt.savefig(paths["out_png"], dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = {
        "site": site,
        "n_cells_total": int(len(grid)),
        "n_cells_classified": int(total_known),
        "counts": {
            "adequate": counts[STATE_ADEQUATE],
            "sunlight_constraint": counts[STATE_SUN_ONLY],
            "ventilation_constraint": counts[STATE_VENT_ONLY],
            "compound_constraint": counts[STATE_COMPOUND],
            "nodata": counts[STATE_NODATA],
        },
        "shares": {
            "adequate": counts[STATE_ADEQUATE] / total_known if total_known else 0.0,
            "sunlight_constraint": counts[STATE_SUN_ONLY] / total_known if total_known else 0.0,
            "ventilation_constraint": counts[STATE_VENT_ONLY] / total_known if total_known else 0.0,
            "compound_constraint": counts[STATE_COMPOUND] / total_known if total_known else 0.0,
        },
        "thresholds": {
            "sun_hours_winter_min": THRESHOLD_SUN_HRS,
            "lambda_f_max": THRESHOLD_LAMBDA_F,
        },
    }
    paths["out_stats"].write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    sh = stats["shares"]
    print(f"\n[{site}] wrote {paths['out_png']}")
    print(f"  n cells: {len(grid)} (classified {total_known})")
    print(f"    adequate:                {counts[STATE_ADEQUATE]:5d}  ({sh['adequate'] * 100:5.1f}%)")
    print(f"    sunlight constraint:     {counts[STATE_SUN_ONLY]:5d}  ({sh['sunlight_constraint'] * 100:5.1f}%)")
    print(f"    ventilation constraint:  {counts[STATE_VENT_ONLY]:5d}  ({sh['ventilation_constraint'] * 100:5.1f}%)")
    print(f"    compound constraint:     {counts[STATE_COMPOUND]:5d}  ({sh['compound_constraint'] * 100:5.1f}%)")
    print(f"    no data:                 {counts[STATE_NODATA]:5d}")
    return stats


def run_site(site: str, study_area: bool = False) -> dict:
    """`study_area=True` (Maré only): filter the grid to the 16-community
    study area (union ∩ data extent; src/brisa_solar/mare_study_area.py)
    before classifying, and write to the `*_study_area` suffixed outputs —
    the default whole-data-extent outputs (used by the technical report,
    manuscript figures, and the project hub) are left untouched."""
    paths = site_paths(site, study_area=study_area)
    for k in ("grid", "solar"):
        if not paths[k].exists():
            raise FileNotFoundError(f"[{site}] missing input: {paths[k]}")
    grid = gpd.read_file(paths["grid"])
    title_suffix = ""
    if study_area:
        from src.brisa_solar import mare_study_area as msa

        sa = msa.load_study_area()
        mask = msa.within_mask(grid["centroid_x"].to_numpy(), grid["centroid_y"].to_numpy(), sa["study_area"])
        grid = grid.loc[mask].reset_index(drop=True)
        title_suffix = " — study area (16 communities ∩ data extent)"
    grid = aggregate_solar_to_cells(grid, paths["solar"])
    state = classify(grid)
    return render(site, grid, state, paths, title_suffix=title_suffix)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--site", choices=SITES, help="single site to process")
    p.add_argument("--all", action="store_true", help="process every site")
    p.add_argument("--study-area", action="store_true",
                    help="Maré only: filter to the 16-community study area "
                    "(union of Redes da Maré's communities ∩ the site data "
                    "extent) instead of the whole bairro; writes to "
                    "*_study_area suffixed outputs, never the default ones")
    args = p.parse_args()

    if not args.site and not args.all:
        p.error("must pass --site <name> or --all")
    if args.study_area and (args.all or args.site != "maré"):
        p.error("--study-area is only defined for --site maré")

    sites = SITES if args.all else [args.site]
    for site in sites:
        run_site(site, study_area=args.study_area)
    return 0


if __name__ == "__main__":
    sys.exit(main())
