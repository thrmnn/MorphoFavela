#!/usr/bin/env python3
"""Cross-site terrain-aspect analysis.

Aspect (terrain orientation) is a first-order driver of pedestrian-level
sun exposure in mountainous topography: in the southern hemisphere
north-facing slopes receive more winter sun. This script reads
``grid_metrics.gpkg`` for one or more sites, runs a per-site OLS of SVF
(used as the per-cell sky-exposure proxy that exists across all sites)
on a (slope, lambda_p, aspect_sin, aspect_cos) covariate set, and
optionally repeats the regression with measured solar hours wherever
``svf_streets_solar.gpkg`` is available.

Output per site
---------------
``outputs/{site}/morphometrics/aspect_regression.csv`` — one row per
response variable (``svf`` always; ``solar_hours`` when measured),
with ``r2``, intercept, and per-predictor coefficients.

``outputs/{site}/morphometrics/aspect_quadrant_summary.csv`` — N/E/S/W
quadrant means and counts of SVF (and solar_hours where available),
restricted to cells with slope ≥ ``--slope-threshold`` (default 5°)
since flat cells have no meaningful aspect.

Usage::

    python scripts/run_aspect_analysis.py --site vidigal
    python scripts/run_aspect_analysis.py --all-sites
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.morphometry.aspect import aspect_quadrant, aspect_to_sincos

logger = logging.getLogger(__name__)

ALL_SITES = ("vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré")
PREDICTORS = ("slope_deg", "lambda_p", "aspect_sin", "aspect_cos")


def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Same minimal OLS as in analyze_cfd_results.py — coef + R²."""
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


def _load_grid(site: str) -> gpd.GeoDataFrame:
    path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "grid" / "grid_metrics.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"grid_metrics.gpkg not found for {site}: {path}")
    return gpd.read_file(path)


def _load_solar_streets(site: str) -> gpd.GeoDataFrame | None:
    path = PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets_solar.gpkg"
    if not path.exists():
        return None
    return gpd.read_file(path)


def _attach_grid_aspect_to_solar(
    solar: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """For each solar street point, look up the containing grid cell's
    aspect_deg, slope_deg, lambda_p. Spatial join is by point-in-cell
    (cells are rectangles; the grid covers the full site)."""
    if not {"aspect_deg", "slope_deg", "lambda_p"}.issubset(grid.columns):
        return solar
    if solar.crs != grid.crs:
        solar = solar.to_crs(grid.crs)
    keep = grid[["geometry", "aspect_deg", "slope_deg", "lambda_p"]]
    return gpd.sjoin(solar, keep, how="left", predicate="within").drop(columns="index_right")


def _per_site_regression(grid: gpd.GeoDataFrame, solar: gpd.GeoDataFrame | None) -> pd.DataFrame:
    rows = []

    sin_a, cos_a = aspect_to_sincos(grid["aspect_deg"].values)
    X_grid = np.column_stack([grid["slope_deg"].values, grid["lambda_p"].values, sin_a, cos_a])

    for response in ("svf",):
        if response not in grid.columns:
            continue
        fit = _ols(grid[response].values.astype(float), X_grid)
        row = {
            "response": response,
            "source": "grid_metrics",
            "n": fit["n"],
            "r2": fit["r2"],
            "intercept": fit["coef"][0],
        }
        for j, p in enumerate(PREDICTORS):
            row[f"coef_{p}"] = fit["coef"][1 + j]
        rows.append(row)

    if solar is not None and not solar.empty and "solar_hours" in solar.columns:
        sin_a, cos_a = aspect_to_sincos(solar["aspect_deg"].values)
        X_sol = np.column_stack([solar["slope_deg"].values, solar["lambda_p"].values, sin_a, cos_a])
        fit = _ols(solar["solar_hours"].values.astype(float), X_sol)
        row = {
            "response": "solar_hours",
            "source": "svf_streets_solar",
            "n": fit["n"],
            "r2": fit["r2"],
            "intercept": fit["coef"][0],
        }
        for j, p in enumerate(PREDICTORS):
            row[f"coef_{p}"] = fit["coef"][1 + j]
        rows.append(row)

    return pd.DataFrame(rows)


def _per_quadrant_summary(
    grid: gpd.GeoDataFrame,
    solar: gpd.GeoDataFrame | None,
    slope_threshold: float,
) -> pd.DataFrame:
    """Mean SVF (and solar_hours when available) per N/E/S/W quadrant.

    Cells with slope < ``slope_threshold`` are excluded — aspect is
    meaningless on flat ground.
    """
    rows = []
    sloped = grid[grid["slope_deg"].fillna(0) >= slope_threshold].copy()
    sloped["quadrant"] = aspect_quadrant(sloped["aspect_deg"].values)
    for q in ("N", "E", "S", "W"):
        sub = sloped[sloped["quadrant"] == q]
        rows.append(
            {
                "source": "grid_metrics",
                "metric": "svf",
                "quadrant": q,
                "n_cells": int(len(sub)),
                "mean": float(sub["svf"].mean()) if len(sub) else np.nan,
                "std": float(sub["svf"].std()) if len(sub) else np.nan,
            }
        )

    if solar is not None and not solar.empty and "solar_hours" in solar.columns:
        sol = solar.copy()
        sol = sol[sol["slope_deg"].fillna(0) >= slope_threshold]
        sol["quadrant"] = aspect_quadrant(sol["aspect_deg"].values)
        for q in ("N", "E", "S", "W"):
            sub = sol[sol["quadrant"] == q]
            rows.append(
                {
                    "source": "svf_streets_solar",
                    "metric": "solar_hours",
                    "quadrant": q,
                    "n_cells": int(len(sub)),
                    "mean": float(sub["solar_hours"].mean()) if len(sub) else np.nan,
                    "std": float(sub["solar_hours"].std()) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def analyse(site: str, slope_threshold: float = 5.0) -> dict:
    out_dir = PROJECT_ROOT / "outputs" / site / "morphometrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = _load_grid(site)
    solar = _load_solar_streets(site)
    if solar is not None:
        solar = _attach_grid_aspect_to_solar(solar, grid)

    reg = _per_site_regression(grid, solar)
    reg_path = out_dir / "aspect_regression.csv"
    reg.to_csv(reg_path, index=False)
    logger.info("Wrote %s (%d rows)", reg_path, len(reg))

    quad = _per_quadrant_summary(grid, solar, slope_threshold=slope_threshold)
    quad_path = out_dir / "aspect_quadrant_summary.csv"
    quad.to_csv(quad_path, index=False)
    logger.info("Wrote %s (%d rows)", quad_path, len(quad))

    return {
        "site": site,
        "regression_rows": len(reg),
        "quadrant_rows": len(quad),
        "has_solar_data": solar is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-site terrain-aspect analysis.")
    parser.add_argument("--site", default=None, help="Single site, e.g. 'vidigal'.")
    parser.add_argument("--all-sites", action="store_true", help=f"Run on all: {ALL_SITES}")
    parser.add_argument(
        "--slope-threshold",
        type=float,
        default=5.0,
        help="Minimum slope (deg) for a cell to participate in quadrant summary.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.all_sites:
        sites = ALL_SITES
    elif args.site:
        sites = (args.site,)
    else:
        parser.error("Pass --site SITE or --all-sites.")

    for site in sites:
        try:
            result = analyse(site, slope_threshold=args.slope_threshold)
            print(
                f"site={result['site']} reg_rows={result['regression_rows']} "
                f"quad_rows={result['quadrant_rows']} "
                f"solar={'yes' if result['has_solar_data'] else 'no'}"
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", site, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
