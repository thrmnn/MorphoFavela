#!/usr/bin/env python3
"""Full-campaign CFD patch allocation — incremental on top of pilot patches.

Loads existing pilot patches as fixed seeds, allocates additional patches
to reach a per-site target using priority-weighted strata, and exports
everything (pilot + new) as a unified campaign dataset.

Usage:
    python scripts/run_campaign_sampling.py                  # all 5 sites
    python scripts/run_campaign_sampling.py --site vidigal   # single site
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Point, box

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import core functions from the pilot sampling script
_spec = importlib.util.spec_from_file_location(
    "pilot_sampling", PROJECT_ROOT / "scripts" / "run_pilot_sampling.py"
)
_pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pilot)

load_and_validate = _pilot.load_and_validate
assign_strata = _pilot.assign_strata
exclude_edges = _pilot.exclude_edges
build_strata_summary = _pilot.build_strata_summary
select_within_stratum = _pilot.select_within_stratum
validate_blocken = _pilot.validate_blocken
export_patch_data = _pilot.export_patch_data
_resolve = _pilot._resolve
_hillshade = _pilot._hillshade
_stratum_cmap = _pilot._stratum_cmap
_STRATUM_COLORS = _pilot._STRATUM_COLORS
_md5 = _pilot._md5
_git_hash = _pilot._git_hash
_safe_float = _pilot._safe_float
SITE_PRESETS = _pilot.SITE_PRESETS
CONFIG = _pilot.CONFIG
PATCH_RADIUS_M = _pilot.PATCH_RADIUS_M

from src.cartography import (  # noqa: E402
    add_north_arrow,
    add_scale_bar,
    add_terrain_contours,
    apply_publication_style,
    format_utm_axes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("campaign_sampling")

# ═══════════════════════════════════════════════════════════════
#  Campaign targets and priority weights
# ═══════════════════════════════════════════════════════════════

CAMPAIGN_TARGETS = {
    "vidigal": 22,
    "rocinha": 25,
    "riodaspedras": 23,
    "complexo_do_alemao": 25,
    "maré": 25,
}

SVF_PRIORITY = {
    "SVF1": 2.0,  # oversample low-SVF (health-relevant)
    "SVF2": 1.0,  # baseline
    "SVF3": 0.8,  # slight undersample
}


# ═══════════════════════════════════════════════════════════════
#  Campaign allocation
# ═══════════════════════════════════════════════════════════════


def allocate_campaign(
    summary: pd.DataFrame,
    existing_per_stratum: dict[str, int],
    target_total: int,
) -> dict[str, int]:
    """Allocate campaign patches with SVF-priority weighting.

    Returns total allocation per stratum (existing + new).
    """
    existing_total = sum(existing_per_stratum.values())
    additional = target_total - existing_total

    if additional <= 0:
        logger.info("Already at or above target (%d >= %d).", existing_total, target_total)
        return existing_per_stratum.copy()

    # Weighted scores: eligible_cells * SVF_priority
    scores: dict[str, float] = {}
    for _, row in summary.iterrows():
        sid = row["stratum_id"]
        if row["n_cells_eligible"] == 0:
            scores[sid] = 0.0
            continue
        svf_bin = sid.split("_")[0]  # SVF1, SVF2, SVF3
        mult = SVF_PRIORITY.get(svf_bin, 1.0)
        scores[sid] = row["n_cells_eligible"] * mult

    total_score = sum(scores.values())
    if total_score == 0:
        return existing_per_stratum.copy()

    # Proportional allocation of additional patches
    allocation: dict[str, int] = {}
    for sid in summary["stratum_id"]:
        existing = existing_per_stratum.get(sid, 0)
        if scores.get(sid, 0) > 0:
            ideal_add = additional * scores[sid] / total_score
            allocation[sid] = existing + max(0, round(ideal_add))
        else:
            allocation[sid] = existing

    # Ensure every populated stratum has at least 1
    for _, row in summary.iterrows():
        sid = row["stratum_id"]
        if row["n_cells_eligible"] > 0 and allocation.get(sid, 0) == 0:
            allocation[sid] = 1

    # Adjust to hit exact target
    current = sum(allocation.values())
    occupied = [
        sid
        for sid in allocation
        if scores.get(sid, 0) > 0 and allocation[sid] > existing_per_stratum.get(sid, 0)
    ]

    while current > target_total and occupied:
        # Remove from stratum with highest ratio of allocation/score
        best = max(occupied, key=lambda s: allocation[s] / max(scores[s], 0.01))
        if allocation[best] > existing_per_stratum.get(best, 0):
            allocation[best] -= 1
            current -= 1
        else:
            occupied.remove(best)

    while current < target_total:
        # Add to stratum with highest score/allocation ratio
        addable = [s for s in allocation if scores.get(s, 0) > 0]
        if not addable:
            break
        best = max(addable, key=lambda s: scores[s] / max(allocation[s], 0.01))
        allocation[best] += 1
        current += 1

    return allocation


# ═══════════════════════════════════════════════════════════════
#  Campaign sampling per site
# ═══════════════════════════════════════════════════════════════


def run_campaign_site(config: dict, target: int) -> gpd.GeoDataFrame | None:
    """Run campaign sampling for a single site."""
    t0 = time.time()
    site = config["site_id"]
    prefix = config["site_prefix"]
    np.random.seed(config["random_seed"])

    logger.info("=" * 60)
    logger.info("CAMPAIGN SAMPLING: %s (target=%d)", site.upper(), target)
    logger.info("=" * 60)

    # 1. Load pilot patches
    pilot_dir = _resolve(config["output_dir"].replace("campaign_sampling", "pilot_sampling"))
    pilot_csv = pilot_dir / "pilot_patches.csv"
    pilot_gpkg = pilot_dir / "pilot_patches.gpkg"

    if not pilot_csv.exists():
        logger.error("Pilot patches not found: %s", pilot_csv)
        return None

    pilot_df = pd.read_csv(pilot_csv)
    n_pilot = len(pilot_df)
    logger.info("Loaded %d pilot patches from %s", n_pilot, pilot_csv)

    # Pilot coordinates for spacing
    pilot_coords = pilot_df[["center_x", "center_y"]].values

    # 2. Load grid, buildings, run stratification + exclusion
    grid, buildings, boundary = load_and_validate(config)
    grid = assign_strata(grid, config)
    eligible, exclusion_stats = exclude_edges(grid, buildings, config)

    # 3. Build summary and campaign allocation
    summary = build_strata_summary(grid, eligible, config)

    # Count existing pilot patches per stratum
    existing_per_stratum: dict[str, int] = {}
    # Need to map pilot patches to strata — match by nearest grid cell
    pilot_strata = []
    for _, prow in pilot_df.iterrows():
        cx, cy = prow["center_x"], prow["center_y"]
        dists = np.sqrt(
            (eligible["centroid_x"] - cx) ** 2 + (eligible["centroid_y"] - cy) ** 2
        )
        if dists.min() < 10:  # within one cell
            nearest_idx = dists.idxmin()
            sid = eligible.loc[nearest_idx, "stratum_id"]
        else:
            # Try full grid
            dists_full = np.sqrt(
                (grid["centroid_x"] - cx) ** 2 + (grid["centroid_y"] - cy) ** 2
            )
            nearest_idx = dists_full.idxmin()
            sid = grid.loc[nearest_idx, "stratum_id"]
        pilot_strata.append(sid)
        existing_per_stratum[sid] = existing_per_stratum.get(sid, 0) + 1

    allocation = allocate_campaign(summary, existing_per_stratum, target)
    summary["n_existing"] = summary["stratum_id"].map(existing_per_stratum).fillna(0).astype(int)
    summary["n_target"] = summary["stratum_id"].map(allocation).fillna(0).astype(int)
    summary["n_additional"] = summary["n_target"] - summary["n_existing"]

    # Print allocation table
    logger.info("")
    logger.info("=" * 90)
    logger.info("CAMPAIGN ALLOCATION")
    logger.info(
        "%-20s  %6s  %6s  %6s  %6s  %6s",
        "Stratum", "Elig.", "Pilot", "Target", "Add", "Bins",
    )
    logger.info("-" * 90)
    for _, r in summary.iterrows():
        bins = f"{r['SVF_bin']} | {r['slope_bin']} | {r['lambda_p_bin']}"
        logger.info(
            "%-20s  %6d  %6d  %6d  %+5d  %s",
            r["stratum_id"],
            r["n_cells_eligible"],
            r["n_existing"],
            r["n_target"],
            r["n_additional"],
            bins,
        )
    total_add = summary["n_additional"].sum()
    logger.info(
        "%-20s  %6d  %6d  %6d  %+5d",
        "TOTAL",
        summary["n_cells_eligible"].sum(),
        n_pilot,
        summary["n_target"].sum(),
        total_add,
    )
    logger.info("=" * 90)

    # 4. Select additional patches with pilot seeds
    min_spacing = config["min_patch_spacing"]
    selected_coords = pilot_coords.copy()

    strata_order = sorted(
        allocation, key=lambda s: allocation.get(s, 0) - existing_per_stratum.get(s, 0), reverse=True
    )

    new_selected: list[gpd.GeoDataFrame] = []
    for sid in strata_order:
        n_add = allocation[sid] - existing_per_stratum.get(sid, 0)
        if n_add <= 0:
            continue

        candidates = eligible[eligible["stratum_id"] == sid].copy()

        # Exclude candidates too close to any existing or already-selected patch
        if len(selected_coords) > 0 and len(candidates) > 0:
            cand_xy = candidates[["centroid_x", "centroid_y"]].values
            dists = cdist(cand_xy, selected_coords)
            too_close = dists.min(axis=1) < min_spacing
            candidates = candidates[~too_close]

        if candidates.empty:
            logger.warning(
                "  %s: 0 candidates after spacing (wanted +%d).", sid, n_add
            )
            continue

        actual = min(n_add, len(candidates))
        picks = select_within_stratum(candidates, actual, config)
        new_selected.append(picks)

        new_xy = picks[["centroid_x", "centroid_y"]].values
        selected_coords = np.vstack([selected_coords, new_xy])
        if actual < n_add:
            logger.warning(
                "  %s: selected %d / %d wanted (candidate shortage).", sid, actual, n_add
            )
        else:
            logger.info("  %s: +%d new patches.", sid, actual)

    # 5. Merge pilot + new
    n_new = sum(len(df) for df in new_selected)
    logger.info("Selected %d new patches (pilot: %d, total: %d).", n_new, n_pilot, n_pilot + n_new)

    # Build new patches GeoDataFrame
    if new_selected:
        new_gdf = gpd.GeoDataFrame(
            pd.concat(new_selected, ignore_index=True), crs=grid.crs
        )
        new_gdf = new_gdf.sort_values(["stratum_id", "centroid_x"]).reset_index(drop=True)
        new_gdf["patch_id"] = [
            f"{prefix}-P{n_pilot + i + 1:02d}" for i in range(len(new_gdf))
        ]
        new_gdf["is_pilot"] = False

        # Blocken validation for new patches
        new_gdf = validate_blocken(new_gdf, buildings, config)
    else:
        new_gdf = gpd.GeoDataFrame()

    # Reconstruct pilot patches as GeoDataFrame with grid cell geometry
    pilot_rows = []
    for i, (_, prow) in enumerate(pilot_df.iterrows()):
        cx, cy = prow["center_x"], prow["center_y"]
        # Find nearest grid cell
        dists = np.sqrt(
            (grid["centroid_x"] - cx) ** 2 + (grid["centroid_y"] - cy) ** 2
        )
        nearest_idx = dists.idxmin()
        row_data = grid.loc[nearest_idx].to_dict()
        row_data["patch_id"] = prow["patch_id"]
        row_data["is_pilot"] = True
        row_data["centroid_x"] = cx
        row_data["centroid_y"] = cy
        row_data["stratum_id"] = pilot_strata[i]
        # Carry over blocken data from pilot
        row_data["H_max_analysis"] = prow.get("H_max_analysis", 0)
        row_data["blocken_radius_required"] = prow.get("blocken_radius_required", 0)
        row_data["blocken_ok"] = True
        row_data["_building_coverage"] = prow.get("building_coverage", 0)
        row_data["_domain_coverage"] = 1.0
        pilot_rows.append(row_data)

    pilot_gdf = gpd.GeoDataFrame(pilot_rows, crs=grid.crs)

    # Merge
    common_cols = sorted(set(pilot_gdf.columns) & set(new_gdf.columns)) if len(new_gdf) > 0 else list(pilot_gdf.columns)
    if "geometry" not in common_cols:
        common_cols.append("geometry")

    if len(new_gdf) > 0:
        all_patches = gpd.GeoDataFrame(
            pd.concat([pilot_gdf[common_cols], new_gdf[common_cols]], ignore_index=True),
            crs=grid.crs,
        )
    else:
        all_patches = pilot_gdf

    # Ensure is_pilot column survives
    if "is_pilot" not in all_patches.columns:
        all_patches["is_pilot"] = True
        if len(new_gdf) > 0:
            all_patches.loc[n_pilot:, "is_pilot"] = False

    # Sort by patch ID
    all_patches = all_patches.sort_values("patch_id").reset_index(drop=True)

    # 6. Write outputs
    out_dir = _resolve(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "patches").mkdir(exist_ok=True)

    r = config["cfd_domain_radius"]

    # GeoPackage — analysis patches (100 m-diameter circles)
    patches_out = all_patches.copy()
    patches_out["geometry"] = [
        Point(row["centroid_x"], row["centroid_y"]).buffer(PATCH_RADIUS_M)
        for _, row in all_patches.iterrows()
    ]
    out_cols = [
        "patch_id", "is_pilot", "centroid_x", "centroid_y", "stratum_id",
        config["col_svf"], config["col_lambda_p"], config["col_slope"],
        config["col_porosity"], config["col_sigma_h"], config["col_h_mean"],
        "H_max_analysis", "blocken_radius_required", "geometry",
    ]
    out_cols = [c for c in out_cols if c in patches_out.columns]
    patches_export = patches_out[out_cols].rename(
        columns={"centroid_x": "center_x", "centroid_y": "center_y"}
    )

    gpkg_path = out_dir / "campaign_patches.gpkg"
    patches_export.to_file(gpkg_path, layer="analysis_patches", driver="GPKG")

    # Domains layer
    domains_out = gpd.GeoDataFrame(
        {"patch_id": all_patches["patch_id"].values, "is_pilot": all_patches["is_pilot"].values},
        geometry=[Point(row["centroid_x"], row["centroid_y"]).buffer(r) for _, row in all_patches.iterrows()],
        crs=grid.crs,
    )
    domains_out.to_file(gpkg_path, layer="cfd_domains", driver="GPKG")
    logger.info("Saved %s", gpkg_path)

    # CSV
    csv_df = patches_export.drop(columns="geometry")
    csv_df.to_csv(out_dir / "campaign_patches.csv", index=False)

    # Stratum summary
    summary.to_csv(out_dir / "stratum_summary.csv", index=False)

    # Sampling log
    log_data = {
        "config": {k: v for k, v in config.items()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "target": target,
        "n_pilot": n_pilot,
        "n_new": n_new,
        "n_total": len(all_patches),
        "excluded_cells": exclusion_stats,
        "strata_allocation": {
            sid: {"existing": existing_per_stratum.get(sid, 0), "target": allocation[sid]}
            for sid in allocation
        },
    }
    with open(out_dir / "sampling_log.json", "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    # 7. Export per-patch data for ALL patches
    logger.info("Exporting per-patch data for %d patches...", len(all_patches))
    # Temporarily set output_dir for export function
    export_config = config.copy()
    export_config["output_dir"] = str(out_dir.relative_to(PROJECT_ROOT))
    export_patch_data(all_patches, buildings, export_config)

    # 8. Figures
    _generate_campaign_figures(grid, all_patches, buildings, config, out_dir)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("CAMPAIGN SAMPLING COMPLETE: %s in %.1f s", site.upper(), elapsed)
    logger.info("  Pilot: %d  New: %d  Total: %d / %d target", n_pilot, n_new, len(all_patches), target)
    logger.info("  Output: %s", out_dir)
    logger.info("=" * 60)

    return all_patches


# ═══════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════


def _generate_campaign_figures(
    grid: gpd.GeoDataFrame,
    all_patches: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    config: dict,
    out_dir: Path,
) -> None:
    """Generate campaign-level figures."""
    import rasterio

    apply_publication_style()
    site_name = config["site_id"].replace("_", " ").title()

    # --- Figure 1: Map ---
    fig, ax = plt.subplots(figsize=(12, 10))

    dtm_path = _resolve(config["dtm"])
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
                ax.imshow(shade, cmap="gray", vmin=0, vmax=1,
                          extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                          alpha=0.4, zorder=0)
        except Exception:
            pass

    # Grid cells by stratum
    cmap = _stratum_cmap(grid["stratum_id"].tolist())
    g = grid[grid["stratum_id"].notna()].copy()
    g["_color"] = g["stratum_id"].map(cmap)
    g.plot(ax=ax, color=g["_color"], alpha=0.4, edgecolor="none", zorder=1)

    if dtm_path.exists():
        add_terrain_contours(ax, dtm_path, alpha=0.2)

    pilots = all_patches[all_patches["is_pilot"] == True]
    new_patches = all_patches[all_patches["is_pilot"] == False]

    # Pilot patches — solid circle, 100 m diameter
    for _, row in pilots.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]
        ax.add_patch(plt.Circle(
            (cx, cy), PATCH_RADIUS_M,
            linewidth=1.2, edgecolor="#1565c0", facecolor="#1565c0", alpha=0.15, zorder=4,
        ))
        ax.plot(cx, cy, "o", color="#1565c0", markersize=4, zorder=6)

    # New patches — dashed circle, 100 m diameter
    for _, row in new_patches.iterrows():
        cx, cy = row["centroid_x"], row["centroid_y"]
        ax.add_patch(plt.Circle(
            (cx, cy), PATCH_RADIUS_M,
            linewidth=1.5, edgecolor="#d62728", facecolor="#d62728", alpha=0.12,
            linestyle="--", zorder=5,
        ))
        ax.plot(cx, cy, "s", color="#d62728", markersize=4, zorder=7)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#1565c0", alpha=0.3, edgecolor="#1565c0", label=f"Pilot ({len(pilots)})"),
        Patch(facecolor="#d62728", alpha=0.3, edgecolor="#d62728", linestyle="--", label=f"New ({len(new_patches)})"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)

    ax.set_title(
        f"Campaign Patches \u2014 {site_name} (n={len(all_patches)})",
        fontsize=14, fontweight="bold",
    )
    format_utm_axes(ax)
    add_scale_bar(ax)
    add_north_arrow(ax)
    ax.set_aspect("equal")

    fig.savefig(out_dir / "figures" / "fig_campaign_patches_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved fig_campaign_patches_map.png")

    # --- Figure 2: Feature space ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    c_svf = config["col_svf"]
    c_lp = config["col_lambda_p"]
    c_slp = config["col_slope"]
    c_sh = config["col_sigma_h"]
    pairs = [
        (c_svf, c_slp, "SVF", "Slope (\u00b0)"),
        (c_svf, c_lp, "SVF", "\u03bbp"),
        (c_slp, c_lp, "Slope (\u00b0)", "\u03bbp"),
        (c_svf, c_sh, "SVF", "\u03c3H (m)"),
    ]

    for ax, (xc, yc, xl, yl) in zip(axes.flat, pairs):
        valid = grid[[xc, yc]].dropna()
        ax.scatter(valid[xc], valid[yc], s=2, c="#cccccc", alpha=0.15, rasterized=True)

        for is_p, color, marker, label in [(True, "#1565c0", "o", "Pilot"), (False, "#d62728", "s", "New")]:
            subset = all_patches[all_patches["is_pilot"] == is_p]
            for _, row in subset.iterrows():
                xv, yv = row.get(xc), row.get(yc)
                if pd.notna(xv) and pd.notna(yv):
                    ax.scatter(xv, yv, s=60, c=color, edgecolors="k", linewidths=0.4,
                               marker=marker, zorder=5)

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.2)

    # Manual legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565c0", markersize=8, label="Pilot"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", markersize=8, label="New"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"Feature-Space Coverage \u2014 {site_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_dir / "figures" / "fig_campaign_patches_featurespace.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved fig_campaign_patches_featurespace.png")

    # --- Figure 3: Context thumbnails for NEW patches only ---
    if len(new_patches) > 0:
        n = len(new_patches)
        ncols = min(5, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes_arr = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
        if n == 1:
            axes_arr = np.array([axes_arr])
        axes_flat = np.atleast_2d(axes_arr).flat

        radius = config["cfd_domain_radius"]
        view_half = 300
        bld_sindex = buildings.sindex

        for i, (_, row) in enumerate(new_patches.iterrows()):
            ax = axes_flat[i]
            cx, cy = row["centroid_x"], row["centroid_y"]
            vbox = box(cx - view_half, cy - view_half, cx + view_half, cy + view_half)
            cands = list(bld_sindex.intersection(vbox.bounds))
            if cands:
                nearby = buildings.iloc[cands]
                nearby = nearby[nearby.intersects(vbox)]
                nearby.plot(ax=ax, facecolor="#d0d0d0", edgecolor="#666666", linewidth=0.3, zorder=1)

            ax.add_patch(plt.Circle(
                (cx, cy), PATCH_RADIUS_M,
                linewidth=1.5, edgecolor="#d62728", facecolor="none", linestyle="--", zorder=5,
            ))
            ax.add_patch(plt.Circle(
                (cx, cy), radius, linewidth=1.0, edgecolor="#1f77b4", facecolor="none", linestyle=":", zorder=4,
            ))

            svf = row.get(config["col_svf"])
            lp = row.get(config["col_lambda_p"])
            slope = row.get(config["col_slope"])
            title = row["patch_id"]
            if pd.notna(svf) and pd.notna(lp):
                title += f" | SVF={svf:.2f}  \u03bbp={lp:.2f}  slope={slope:.0f}\u00b0"
            ax.set_title(title, fontsize=7, fontweight="bold")
            ax.set_xlim(cx - view_half, cx + view_half)
            ax.set_ylim(cy - view_half, cy + view_half)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(i + 1, nrows * ncols):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"New Patch Context \u2014 {site_name}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / "fig_campaign_patches_context.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved fig_campaign_patches_context.png")


# ═══════════════════════════════════════════════════════════════
#  Cross-site summary
# ═══════════════════════════════════════════════════════════════


def generate_cross_site_summary(all_results: dict[str, gpd.GeoDataFrame]) -> None:
    """Generate campaign-level summary across all sites."""
    out_dir = PROJECT_ROOT / "outputs" / "comparative" / "final_allocation"
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_publication_style()

    # Master allocation table
    master_rows = []
    for site, patches in all_results.items():
        if patches is None:
            continue
        for _, row in patches.iterrows():
            master_rows.append({
                "patch_id": row["patch_id"],
                "site": site,
                "is_pilot": row.get("is_pilot", True),
                "stratum_id": row.get("stratum_id", ""),
                "center_x": row["centroid_x"],
                "center_y": row["centroid_y"],
                "svf": row.get("svf"),
                "lambda_p": row.get("lambda_p"),
                "slope_deg": row.get("slope_deg"),
                "sigma_h": row.get("sigma_h"),
                "H_mean": row.get("H_mean"),
                "porosity": row.get("porosity"),
            })

    master = pd.DataFrame(master_rows)
    master.to_csv(out_dir / "campaign_allocation_table.csv", index=False)
    logger.info("Saved campaign_allocation_table.csv (%d patches)", len(master))

    # Strata summary across sites
    strata_rows = []
    for site in all_results:
        csv_path = PROJECT_ROOT / "outputs" / site / "sampling_cfd" / "campaign_sampling" / "stratum_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["site"] = site
            strata_rows.append(df)

    if strata_rows:
        strata_all = pd.concat(strata_rows, ignore_index=True)
        pivot = strata_all.pivot_table(
            index="stratum_id", columns="site", values="n_target", fill_value=0
        ).astype(int)
        pivot["TOTAL"] = pivot.sum(axis=1)
        pivot.to_csv(out_dir / "campaign_strata_summary.csv")
        logger.info("Saved campaign_strata_summary.csv")

    # Cross-site feature space plot
    SITE_COLORS = {
        "vidigal": "#e6194b", "rocinha": "#3cb44b", "riodaspedras": "#4363d8",
        "complexo_do_alemao": "#f58231", "maré": "#911eb4",
    }
    SITE_MARKERS = {
        "vidigal": "o", "rocinha": "s", "riodaspedras": "D",
        "complexo_do_alemao": "^", "maré": "v",
    }

    pairs = [
        ("svf", "slope_deg", "SVF", "Slope (\u00b0)"),
        ("svf", "lambda_p", "SVF", "\u03bbp"),
        ("slope_deg", "lambda_p", "Slope (\u00b0)", "\u03bbp"),
        ("svf", "sigma_h", "SVF", "\u03c3H (m)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (xc, yc, xl, yl) in zip(axes.flat, pairs):
        for site, patches in all_results.items():
            if patches is None:
                continue
            color = SITE_COLORS.get(site, "#333")
            marker = SITE_MARKERS.get(site, "o")

            pilots = patches[patches["is_pilot"] == True]
            new = patches[patches["is_pilot"] == False]

            for subset, alpha, size in [(pilots, 0.6, 60), (new, 0.9, 80)]:
                vals = subset[[xc, yc]].dropna()
                if not vals.empty:
                    ax.scatter(vals[xc], vals[yc], s=size, c=color, marker=marker,
                               edgecolors="k", linewidths=0.3, alpha=alpha, zorder=5)

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.15)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=SITE_MARKERS[s], color="w", markerfacecolor=SITE_COLORS[s],
               markersize=8, label=s.replace("_", " ").title())
        for s in all_results if all_results[s] is not None
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Campaign Feature-Space Coverage \u2014 {len(master)} Patches", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_dir / "fig_campaign_cross_site_featurespace.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig_campaign_cross_site_featurespace.png")

    # Allocation bar chart: patches per stratum colored by site
    if strata_rows:
        fig, ax = plt.subplots(figsize=(14, 6))
        strata_ids = sorted(pivot.index)
        x = np.arange(len(strata_ids))
        bottoms = np.zeros(len(strata_ids))
        site_order = [s for s in SITE_COLORS if s in pivot.columns]

        for site in site_order:
            vals = [pivot.loc[sid, site] if site in pivot.columns else 0 for sid in strata_ids]
            ax.bar(x, vals, bottom=bottoms, label=site.replace("_", " ").title(),
                   color=SITE_COLORS[site], edgecolor="white", linewidth=0.5)
            bottoms += np.array(vals)

        # Total labels on top
        for i, sid in enumerate(strata_ids):
            total = pivot.loc[sid, "TOTAL"]
            ax.text(i, total + 0.3, str(total), ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(strata_ids, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Patches")
        ax.set_title("Campaign Allocation by Stratum", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, ncol=5)
        ax.grid(axis="y", alpha=0.2)

        fig.tight_layout()
        fig.savefig(out_dir / "fig_campaign_allocation_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig_campaign_allocation_summary.png")

    # Campaign report
    n_total = len(master)
    n_pilot = master["is_pilot"].sum()
    n_new = n_total - n_pilot

    # Min inter-patch distance per site
    spacing_info = []
    for site, patches in all_results.items():
        if patches is None or len(patches) < 2:
            continue
        coords = patches[["centroid_x", "centroid_y"]].values
        dm = cdist(coords, coords)
        np.fill_diagonal(dm, np.inf)
        min_d = dm.min()
        spacing_info.append(f"  {site}: {min_d:.0f}m")

    report_lines = [
        "# Campaign Allocation Report",
        "",
        f"**Total patches:** {n_total} ({n_pilot} pilot + {n_new} new)",
        f"**Sites:** {len(all_results)}",
        "",
        "## Per-Site Summary",
        "",
        "| Site | Pilot | New | Total | Target | Shortfall |",
        "|------|-------|-----|-------|--------|-----------|",
    ]

    for site in CAMPAIGN_TARGETS:
        target = CAMPAIGN_TARGETS[site]
        patches = all_results.get(site)
        if patches is None:
            report_lines.append(f"| {site} | — | — | — | {target} | FAILED |")
            continue
        n_p = (patches["is_pilot"] == True).sum()
        n_n = (patches["is_pilot"] == False).sum()
        n_t = len(patches)
        shortfall = target - n_t
        sf_str = f"-{shortfall}" if shortfall > 0 else "0"
        report_lines.append(f"| {site} | {n_p} | {n_n} | {n_t} | {target} | {sf_str} |")

    report_lines.extend([
        "",
        "## Minimum Inter-Patch Distance",
        "",
    ] + spacing_info + [
        "",
        "## Files",
        "",
        "- `campaign_allocation_table.csv` — master table (all patches, all sites)",
        "- `campaign_strata_summary.csv` — 12 strata x 5 sites allocation",
        "- `fig_campaign_cross_site_featurespace.png` — feature-space coverage",
        "- `fig_campaign_allocation_summary.png` — allocation bar chart by stratum",
        "",
        "Per-site outputs at `outputs/{site}/sampling_cfd/campaign_sampling/`.",
    ])

    with open(out_dir / "campaign_report.md", "w") as f:
        f.write("\n".join(report_lines))
    logger.info("Saved campaign_report.md")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Full-campaign CFD patch allocation (incremental on pilot patches)",
    )
    parser.add_argument(
        "--site",
        default=None,
        choices=list(CAMPAIGN_TARGETS.keys()),
        help="Run for a single site (default: all 5)",
    )
    args = parser.parse_args()

    sites = [args.site] if args.site else list(CAMPAIGN_TARGETS.keys())
    all_results: dict[str, gpd.GeoDataFrame | None] = {}

    for site in sites:
        config = CONFIG.copy()
        if site in SITE_PRESETS:
            config.update(SITE_PRESETS[site])

        # Use extended buildings and DTM
        ext_bld = PROJECT_ROOT / "data" / site / "buildings_extended_300m.gpkg"
        ext_dtm = PROJECT_ROOT / "data" / site / "dtm_extended_300m.tif"
        if ext_bld.exists():
            config["buildings"] = str(ext_bld.relative_to(PROJECT_ROOT))
        if ext_dtm.exists():
            config["dtm"] = str(ext_dtm.relative_to(PROJECT_ROOT))

        # Set output to campaign_sampling
        config["output_dir"] = f"outputs/{site}/sampling_cfd/campaign_sampling"

        target = CAMPAIGN_TARGETS[site]
        result = run_campaign_site(config, target)
        all_results[site] = result

    # Cross-site summary
    if len(all_results) > 1:
        generate_cross_site_summary(all_results)


if __name__ == "__main__":
    main()
