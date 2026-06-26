#!/usr/bin/env python3
"""End-to-end analysis pipeline for returned CFD results.

Consumes one site's CFD outputs (auto-detects MorphoFavela-native CSV or
Airflow-native parquet layout, see ``src/cfd_integration/io.py``) and
emits paper-ready artefacts:

* ``per_patch_indicators.csv``  — one row per patch with annualised
  indicators (mean U, p10, stagnation fraction, TKE/TI) joined with
  the patch's morphometric covariates.
* ``grid_with_cfd.gpkg``        — per-cell annualised indicators for
  cells inside each patch's 100 m analysis circle. The unit of
  defensible CFD analysis is the patch (Blocken 2015), but cell-level
  indicators feed Figure 5 maps.
* ``predictor_regression.csv``  — within-site OLS of annual indicators
  against (SVF, λp, slope, σ_h). v0 — diagnostic, not paper-ready.
* ``figures/fig5_wind_panel.png`` — per-cell annual U_mean map; locks
  the spec for Figure 5's wind-velocity panel.
* ``coverage.json``             — which patches/directions returned;
  what was missing.

Usage:
    python scripts/analyze_cfd_results.py --site vidigal

The chain runs against synthetic data today; the same command will
run against real returned results once VDG-P07 lands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cfd_integration import (
    aggregate_to_grid,
    aggregate_to_patch,
    load_campaign_results,
    load_wind_rose,
    weighted_by_wind_rose,
)
from src.cfd_integration.schema import WIND_DIRECTIONS_8
from src.morphometry.aspect import (
    aspect_to_sincos,
    aspect_wind_alignment,
    dominant_wind_direction_deg,
)

logger = logging.getLogger(__name__)

# Linear OLS predictors. aspect_deg is circular and is included as the
# orthogonal pair (aspect_sin, aspect_cos); aspect_wind_alignment is the
# scalar cosine of the angle between cell aspect and the dominant wind
# direction (computed from the per-site wind rose).
PREDICTOR_COLS = (
    "svf",
    "lambda_p",
    "slope_deg",
    "sigma_h",
    "aspect_sin",
    "aspect_cos",
    "aspect_wind_alignment",
)
INDICATOR_COLS = ("annual_U_mean", "annual_U_p10", "annual_stagnation_frac", "annual_TI_mean")


def _patch_centers(site: str) -> pd.DataFrame:
    """Load campaign_patches.csv as the source of truth for centers + covariates."""
    csv = (
        PROJECT_ROOT
        / "outputs"
        / site
        / "sampling_cfd"
        / "campaign_sampling"
        / "campaign_patches.csv"
    )
    if not csv.exists():
        raise FileNotFoundError(f"campaign_patches.csv not found for {site}: {csv}")
    return pd.read_csv(csv)


def _site_grid(site: str) -> gpd.GeoDataFrame:
    gpkg = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"grid_metrics.gpkg not found for {site}: {gpkg}")
    return gpd.read_file(gpkg)


def _wind_rose_path(site: str) -> Path:
    return PROJECT_ROOT / "data" / site / "wind_rose.json"


def _annualise_per_patch(
    campaign,
    patches_df: pd.DataFrame,
    wind_rose,
    weight_by: str = "freq_speed",
) -> tuple[pd.DataFrame, dict]:
    """For each returned patch, aggregate per-direction then weighted-combine.

    Returns (per_patch_df, coverage_dict). per_patch_df is indexed by
    patch_id with annual_* + n_directions columns merged onto patches_df.
    """
    centers = patches_df.set_index("patch_id")
    rows = []
    coverage: dict = {"per_patch_directions": {}, "missing_patches": []}
    for patch_id in centers.index:
        if patch_id not in campaign.patches:
            coverage["missing_patches"].append(patch_id)
            continue
        per_direction = campaign.patches[patch_id]
        coverage["per_patch_directions"][patch_id] = sorted(per_direction.keys())
        patch_center = (centers.loc[patch_id, "center_x"], centers.loc[patch_id, "center_y"])

        per_dir_metrics = {
            d: aggregate_to_patch(pr, patch_center) for d, pr in per_direction.items()
        }
        annual = weighted_by_wind_rose(per_dir_metrics, wind_rose=wind_rose, weight_by=weight_by)
        annual["patch_id"] = patch_id
        rows.append(annual)

    if not rows:
        return pd.DataFrame(), coverage

    annual_df = pd.DataFrame(rows).set_index("patch_id")
    merged = centers.join(annual_df, how="inner")
    merged = merged.reset_index()
    return merged, coverage


def _annualise_per_cell(
    campaign,
    patches_df: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    wind_rose,
) -> gpd.GeoDataFrame:
    """For each patch, aggregate cells per-direction then weight across.

    Each cell appears at most once (one cell belongs to one patch). For
    cells in a patch's analysis circle, we annualise by weighted average
    of the per-direction CFD metrics using wind-rose frequencies.
    """
    centers = patches_df.set_index("patch_id")
    cell_frames = []

    if wind_rose is None:
        weights_by_dir = None
    else:
        f = np.array([wind_rose.frequencies.get(d, 0.0) for d in WIND_DIRECTIONS_8])
        u = np.array([wind_rose.mean_speeds.get(d, 1.0) for d in WIND_DIRECTIONS_8])
        raw = f * u
        weights_by_dir = (
            dict(zip(WIND_DIRECTIONS_8, raw / raw.sum()))
            if raw.sum() > 0
            else dict(zip(WIND_DIRECTIONS_8, np.full(8, 1.0 / 8)))
        )

    metric_cols = (
        "cfd_U_mean",
        "cfd_U_p10",
        "cfd_stagnation_frac",
        "cfd_TKE_mean",
        "cfd_TI_mean",
        "cfd_ach",
    )

    for patch_id in centers.index:
        if patch_id not in campaign.patches:
            continue
        per_direction = campaign.patches[patch_id]
        if not per_direction:
            continue
        patch_center = (centers.loc[patch_id, "center_x"], centers.loc[patch_id, "center_y"])

        per_dir_cells: dict[str, gpd.GeoDataFrame] = {}
        for direction, pr in per_direction.items():
            cells = aggregate_to_grid(pr, grid, patch_center)
            if not cells.empty:
                per_dir_cells[direction] = cells

        if not per_dir_cells:
            continue

        first_dir = next(iter(per_dir_cells))
        accum = per_dir_cells[first_dir][["zone_id", "geometry", "centroid_x", "centroid_y"]].copy()
        for col in metric_cols:
            weighted_sum = np.zeros(len(accum))
            weight_total = np.zeros(len(accum))
            for direction, cells in per_dir_cells.items():
                w = weights_by_dir[direction] if weights_by_dir else 1.0 / len(per_dir_cells)
                values = cells[col].values
                valid = ~np.isnan(values)
                weighted_sum[valid] += values[valid] * w
                weight_total[valid] += w
            with np.errstate(invalid="ignore", divide="ignore"):
                annual_values = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)
            accum[f"annual_{col}"] = annual_values

        accum["cfd_patch_id"] = patch_id
        accum["n_directions"] = len(per_dir_cells)
        cell_frames.append(accum)

    if not cell_frames:
        return gpd.GeoDataFrame(geometry=[], crs=grid.crs)

    return gpd.GeoDataFrame(pd.concat(cell_frames, ignore_index=True), crs=grid.crs)


def _enrich_patches_with_aspect(
    patches_df: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    wind_rose,
    radius_m: float = 50.0,
) -> pd.DataFrame:
    """Attach circular-mean aspect + sin/cos + wind-alignment per patch.

    Aspect is sampled by averaging ``aspect_deg`` (circular mean) over
    grid cells whose centroids lie within ``radius_m`` of the patch
    center — i.e. the same 100 m analysis circle used for cell-level
    annualised indicators. The dominant wind direction is the
    frequency-weighted circular mean of the per-site wind rose.

    Returns ``patches_df`` augmented with columns:
        - ``aspect_deg`` (per-patch circular mean over its analysis disk)
        - ``aspect_sin``, ``aspect_cos`` (orthogonal OLS encoding)
        - ``aspect_wind_alignment`` (cos angle vs dominant wind dir)

    Patches missing terrain coverage get NaN; OLS drops them.
    """
    if "aspect_deg" not in grid.columns:
        logger.warning("grid_metrics has no aspect_deg column; skipping aspect enrichment")
        return patches_df

    out = patches_df.copy()
    centroids = grid.geometry.centroid
    cx = centroids.x.values
    cy = centroids.y.values
    aspect_vals = grid["aspect_deg"].values

    aspects = np.full(len(out), np.nan)
    for i, row in enumerate(out.itertuples()):
        dx = cx - row.center_x
        dy = cy - row.center_y
        mask = (dx * dx + dy * dy) <= radius_m * radius_m
        if not mask.any():
            continue
        a = aspect_vals[mask]
        a = a[~np.isnan(a)]
        if a.size == 0:
            continue
        rad = np.deg2rad(a)
        sx = np.sin(rad).mean()
        cxm = np.cos(rad).mean()
        aspects[i] = float(np.rad2deg(np.arctan2(sx, cxm)) % 360.0)

    out["aspect_deg"] = aspects
    sin_a, cos_a = aspect_to_sincos(aspects)
    out["aspect_sin"] = sin_a
    out["aspect_cos"] = cos_a

    wind_dir = dominant_wind_direction_deg(wind_rose) if wind_rose is not None else None
    if wind_dir is None:
        out["aspect_wind_alignment"] = np.nan
    else:
        out["aspect_wind_alignment"] = aspect_wind_alignment(aspects, wind_dir)
    return out


def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Tiny OLS via numpy. Returns coef + R² + residual std.

    For v0 paper-figure prep this is enough; statsmodels can replace it
    when t-stats and confidence intervals matter (Phase 7 manuscript).
    """
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]
    if len(y) < X.shape[1] + 1:
        return {"coef": np.full(X.shape[1] + 1, np.nan), "r2": np.nan, "n": int(len(y))}
    Xb = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    y_hat = Xb @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"coef": coef, "r2": r2, "n": int(len(y))}


def _predictor_regression(per_patch_df: pd.DataFrame) -> pd.DataFrame:
    """Within-site OLS of each annual indicator against the predictor set."""
    available = [c for c in PREDICTOR_COLS if c in per_patch_df.columns]
    X = per_patch_df[list(available)].values
    rows = []
    for indicator in INDICATOR_COLS:
        if indicator not in per_patch_df.columns:
            continue
        y = per_patch_df[indicator].values
        fit = _ols(y, X)
        row = {"indicator": indicator, "n": fit["n"], "r2": fit["r2"], "intercept": fit["coef"][0]}
        for j, predictor in enumerate(available):
            row[f"coef_{predictor}"] = fit["coef"][1 + j]
        rows.append(row)
    return pd.DataFrame(rows)


def _figure_5_wind_panel(
    grid_with_cfd: gpd.GeoDataFrame,
    site: str,
    out_path: Path,
    column: str = "annual_cfd_U_mean",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "paper_figures"))
    try:
        import fig_style  # type: ignore

        fig_style.apply_style()
        figsize = fig_style.SIZE_SINGLE
    except Exception:
        figsize = (4.0, 3.4)

    fig, ax = plt.subplots(figsize=figsize)
    if grid_with_cfd.empty or column not in grid_with_cfd.columns:
        ax.text(0.5, 0.5, "no CFD coverage", ha="center", va="center", transform=ax.transAxes)
    else:
        grid_with_cfd.plot(
            column=column,
            ax=ax,
            cmap="viridis",
            legend=True,
            legend_kwds={"shrink": 0.6, "label": "annual mean U (m/s)"},
            missing_kwds={"color": "lightgrey"},
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{site}: annualised CFD wind speed", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyse(
    site: str,
    out_root: Optional[Path] = None,
    weight_by: str = "freq_speed",
) -> dict:
    """Run the end-to-end pipeline. Returns the coverage report."""
    out_root = (
        Path(out_root) if out_root is not None else PROJECT_ROOT / "outputs" / site / "cfd_analysis"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    figures_dir = out_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    patches_df = _patch_centers(site)
    grid = _site_grid(site)
    wind_rose_path = _wind_rose_path(site)
    if not wind_rose_path.exists():
        raise FileNotFoundError(f"wind_rose.json not found for {site}: {wind_rose_path}")
    wind_rose = load_wind_rose(wind_rose_path, site)
    if abs(sum(wind_rose.frequencies.values()) - 1.0) > 0.05:
        logger.warning(
            "%s wind-rose frequencies sum to %.3f (expected ~1.0)",
            site,
            sum(wind_rose.frequencies.values()),
        )

    patches_df = _enrich_patches_with_aspect(patches_df, grid, wind_rose)

    campaign = load_campaign_results(
        site, results_root=PROJECT_ROOT / "data" / site / "cfd_results"
    )
    logger.info(
        "%s: %d patches returned, %d total simulations",
        site,
        len(campaign.patches),
        sum(len(v) for v in campaign.patches.values()),
    )

    per_patch_df, coverage = _annualise_per_patch(
        campaign, patches_df, wind_rose, weight_by=weight_by
    )
    coverage["site"] = site
    coverage["n_patches_expected"] = int(len(patches_df))
    coverage["n_patches_returned"] = int(len(per_patch_df))
    coverage["weight_method"] = weight_by

    per_patch_path = out_root / "per_patch_indicators.csv"
    per_patch_df.to_csv(per_patch_path, index=False)
    logger.info("Wrote %s (%d rows)", per_patch_path, len(per_patch_df))

    if not per_patch_df.empty:
        regression_df = _predictor_regression(per_patch_df)
        regression_path = out_root / "predictor_regression.csv"
        regression_df.to_csv(regression_path, index=False)
        logger.info("Wrote %s (%d indicators)", regression_path, len(regression_df))

    grid_with_cfd = _annualise_per_cell(campaign, patches_df, grid, wind_rose)
    if not grid_with_cfd.empty:
        grid_path = out_root / "grid_with_cfd.gpkg"
        grid_with_cfd.to_file(grid_path, driver="GPKG")
        logger.info("Wrote %s (%d cells)", grid_path, len(grid_with_cfd))

    _figure_5_wind_panel(grid_with_cfd, site, figures_dir / "fig5_wind_panel.png")
    logger.info("Wrote %s", figures_dir / "fig5_wind_panel.png")

    coverage_path = out_root / "coverage.json"
    with open(coverage_path, "w") as f:
        json.dump(coverage, f, indent=2)
    logger.info("Wrote %s", coverage_path)

    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse returned CFD results for one site.")
    parser.add_argument("--site", required=True, help="Site name (e.g., 'vidigal').")
    parser.add_argument(
        "--weight-by",
        choices=("frequency", "freq_speed", "uniform"),
        default="freq_speed",
        help="Wind-rose weighting method.",
    )
    parser.add_argument("--out", default=None, help="Output root override.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    coverage = analyse(
        site=args.site,
        out_root=Path(args.out) if args.out else None,
        weight_by=args.weight_by,
    )
    print(
        f"site={coverage['site']} "
        f"patches={coverage['n_patches_returned']}/{coverage['n_patches_expected']} "
        f"weight={coverage['weight_method']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
