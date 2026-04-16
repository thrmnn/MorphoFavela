"""Spatial aggregation: map CFD sample points onto the 10m morphometric grid.

Primary path: aggregate only within the 100m-diameter circular analysis patch
of each CFD simulation. Points outside this patch (but still within the 250m
CFD domain) are optionally available via aggregate_to_domain() for robustness
checks.

The 100m-diameter analysis patch is the only zone where CFD results are
scientifically defensible for quantitative use (Blocken 2015, COST Action 732) —
the buffer beyond is atmospheric context to develop realistic turbulence, not
a measurement region.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from src.cfd_integration.metrics import (
    ach,
    low_wind_percentile,
    stagnation_fraction,
    turbulent_intensity,
)
from src.cfd_integration.schema import CFDPatchResult

logger = logging.getLogger(__name__)


def aggregate_to_patch(
    patch_result: CFDPatchResult,
    patch_center_xy: tuple[float, float],
    analysis_patch_diameter: float = 100.0,
    canopy_height: Optional[float] = None,
    stagnation_threshold: float = 0.5,
) -> dict:
    """Aggregate CFD sample points to a single patch-level summary.

    Crops samples to the 100m-diameter circular analysis patch, computes
    scalar metrics.

    Parameters
    ----------
    patch_result : CFDPatchResult
        CFD output for one patch × one wind direction.
    patch_center_xy : (float, float)
        Patch center in UTM (from campaign_patches.csv).
    analysis_patch_diameter : float
        Diameter of the circular analysis zone (default 100m).
    canopy_height : float, optional
        Urban canopy height for ACH computation (m). If None, ACH is skipped.
    stagnation_threshold : float
        |U| below this is 'stagnant' (m/s).

    Returns
    -------
    dict with per-patch metrics.
    """
    cx, cy = patch_center_xy
    radius = analysis_patch_diameter / 2
    r2 = radius * radius

    samples = patch_result.samples
    dx = samples["x"] - cx
    dy = samples["y"] - cy
    mask = (dx * dx + dy * dy) <= r2
    in_patch = samples[mask]

    if in_patch.empty:
        logger.warning(
            "No CFD samples in analysis patch for %s/%s",
            patch_result.metadata.patch_id,
            patch_result.metadata.wind_direction,
        )
        return {
            "patch_id": patch_result.metadata.patch_id,
            "wind_direction": patch_result.metadata.wind_direction,
            "n_samples": 0,
        }

    u_mag = in_patch["U_mag"].values
    tke = in_patch["TKE"].values
    u_ref = patch_result.metadata.wind_speed_ref

    result = {
        "patch_id": patch_result.metadata.patch_id,
        "wind_direction": patch_result.metadata.wind_direction,
        "wind_speed_ref": u_ref,
        "n_samples": len(in_patch),
        "U_mean": float(np.mean(u_mag)),
        "U_median": float(np.median(u_mag)),
        "U_p10": low_wind_percentile(u_mag, 10),
        "U_p90": low_wind_percentile(u_mag, 90),
        "stagnation_frac": stagnation_fraction(u_mag, stagnation_threshold),
        "TKE_mean": float(np.mean(tke)),
        "TI_mean": float(np.mean(turbulent_intensity(tke, u_ref))) if u_ref > 0 else np.nan,
        "vent_efficiency": float(np.mean(u_mag) / u_ref) if u_ref > 0 else np.nan,
    }

    if canopy_height is not None and canopy_height > 0:
        # Per-cell ACH would need cell-level decomposition; here give patch-level
        patch_area = math.pi * radius * radius
        result["ach_patch"] = ach(u_mag, canopy_height, patch_area)

    return result


def aggregate_to_grid(
    patch_result: CFDPatchResult,
    grid: gpd.GeoDataFrame,
    patch_center_xy: tuple[float, float],
    analysis_patch_diameter: float = 100.0,
    stagnation_threshold: float = 0.5,
    canopy_height_col: str = "H_mean",
) -> gpd.GeoDataFrame:
    """Map CFD samples to individual 10m grid cells within the analysis patch.

    For each 10m cell whose centroid lies inside the 100m-diameter circular
    analysis zone, compute metrics from CFD samples falling inside the cell's
    bounding box.

    Parameters
    ----------
    patch_result : CFDPatchResult
    grid : GeoDataFrame
        The site's 10m morphometric grid (must include `centroid_x`, `centroid_y`,
        and optionally `canopy_height_col` for per-cell ACH).
    patch_center_xy : (float, float)
        UTM coords of the patch center.
    analysis_patch_diameter : float
        Diameter of the circular analysis zone (m).

    Returns
    -------
    GeoDataFrame
        Subset of grid (only cells whose centroid is inside the analysis patch)
        with new columns:
            cfd_U_mean, cfd_U_p10, cfd_stagnation_frac, cfd_TKE_mean,
            cfd_TI_mean, cfd_ach, cfd_n_samples
    """
    cx, cy = patch_center_xy
    radius = analysis_patch_diameter / 2
    r2 = radius * radius

    # Select cells whose centroids fall inside the circular analysis patch
    dx = grid["centroid_x"] - cx
    dy = grid["centroid_y"] - cy
    cell_mask = (dx * dx + dy * dy) <= r2
    cells = grid[cell_mask].copy()
    if cells.empty:
        return cells

    samples = patch_result.samples
    u_ref = patch_result.metadata.wind_speed_ref

    # For each cell, aggregate samples within it
    # Assumes cells are ~10m; use a 10m bounding box around each centroid
    cell_size = 10.0
    half_c = cell_size / 2

    metrics = {
        "cfd_U_mean": [], "cfd_U_p10": [], "cfd_stagnation_frac": [],
        "cfd_TKE_mean": [], "cfd_TI_mean": [], "cfd_ach": [],
        "cfd_n_samples": [],
    }

    for _, row in cells.iterrows():
        ccx, ccy = row["centroid_x"], row["centroid_y"]
        in_cell = samples[
            (samples["x"] >= ccx - half_c) & (samples["x"] <= ccx + half_c)
            & (samples["y"] >= ccy - half_c) & (samples["y"] <= ccy + half_c)
        ]
        n = len(in_cell)
        metrics["cfd_n_samples"].append(n)

        if n == 0:
            for k in ["cfd_U_mean", "cfd_U_p10", "cfd_stagnation_frac",
                      "cfd_TKE_mean", "cfd_TI_mean", "cfd_ach"]:
                metrics[k].append(np.nan)
            continue

        u_mag = in_cell["U_mag"].values
        tke = in_cell["TKE"].values
        metrics["cfd_U_mean"].append(float(np.mean(u_mag)))
        metrics["cfd_U_p10"].append(low_wind_percentile(u_mag, 10))
        metrics["cfd_stagnation_frac"].append(stagnation_fraction(u_mag, stagnation_threshold))
        metrics["cfd_TKE_mean"].append(float(np.mean(tke)))
        ti = turbulent_intensity(tke, u_ref) if u_ref > 0 else np.array([np.nan])
        metrics["cfd_TI_mean"].append(float(np.mean(ti)))

        h_canopy = row.get(canopy_height_col, np.nan)
        if pd.notna(h_canopy) and h_canopy > 0:
            metrics["cfd_ach"].append(ach(u_mag, float(h_canopy), cell_size**2))
        else:
            metrics["cfd_ach"].append(np.nan)

    for k, v in metrics.items():
        cells[k] = v

    # Tag with patch and direction
    cells["cfd_patch_id"] = patch_result.metadata.patch_id
    cells["cfd_wind_direction"] = patch_result.metadata.wind_direction
    cells["cfd_wind_speed_ref"] = u_ref

    return cells


def aggregate_to_domain(
    patch_result: CFDPatchResult,
    patch_center_xy: tuple[float, float],
    domain_radius: float = 250.0,
    stagnation_threshold: float = 0.5,
) -> dict:
    """Supplementary: aggregate all samples within the full 250m domain.

    Use only for robustness checks — the buffer zone near the domain
    boundary is biased by BC treatment and should not drive conclusions.
    """
    cx, cy = patch_center_xy
    samples = patch_result.samples
    dist = np.sqrt((samples["x"] - cx)**2 + (samples["y"] - cy)**2)
    in_domain = samples[dist <= domain_radius]

    if in_domain.empty:
        return {"n_samples_domain": 0}

    u_mag = in_domain["U_mag"].values
    return {
        "n_samples_domain": len(in_domain),
        "U_mean_domain": float(np.mean(u_mag)),
        "stagnation_frac_domain": stagnation_fraction(u_mag, stagnation_threshold),
    }
