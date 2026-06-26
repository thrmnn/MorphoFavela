"""Feature substrate for the morpho-signature track (WS-0).

Joins the four un-joined per-site morphometric files into two *linked* tables
that respect the street/grid change-of-support problem (see
``docs/morpho_signature_plan.md``):

- **Fabric table** (areal, per 10 m cell): cell-native morphometrics only. The
  street-aggregated ``svf``/``svf_count`` columns that ``run_morphometric_audit``
  writes into the grid are deliberately *excluded* from the fabric feature set —
  they are experienced outcomes, not fabric inputs, and double-count
  (porosity ≈ f(λp), SVF ≈ f(λp, H/W)).
- **Experience table** (network, per observer): SVF / solar / openness / H-W at
  the pedestrian sample points, each enriched with the fabric covariates of its
  containing cell. Transfer is areal→point (a clean point-in-polygon lookup), the
  legitimate direction; never point→area by mean.

When street conditions must be shown per cell, ``summarize_streets_to_cells``
produces *support-aware* summaries (distributional, not a single mean) with an
explicit support count and a ``has_street_support`` flag; cells with no street
observation stay NULL rather than being filled by nearest-k interpolation.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from src.morphometry.invariants import built_mask

# Cell-native fabric columns. svf / svf_count are intentionally NOT here.
FABRIC_COLS = [
    "zone_id", "zone_area", "lambda_p", "far", "sigma_h", "H_mean",
    "building_count",
    "lambda_f_N", "lambda_f_NE", "lambda_f_E", "lambda_f_SE",
    "lambda_f_S", "lambda_f_SW", "lambda_f_W", "lambda_f_NW",
    "lambda_f_mean", "lambda_f_max",
    "porosity", "slope_deg", "aspect_deg", "street_orientation_entropy",
    "centroid_x", "centroid_y",
]

# Fabric covariates carried onto each street observation (areal→point).
FABRIC_COVARIATES = [
    "lambda_p", "lambda_f_mean", "porosity", "slope_deg",
    "aspect_deg", "H_mean", "sigma_h",
]

SUN_THRESHOLD_H = 2.0  # WHO winter-solstice floor
DEEP_CANYON_CLASS = "DEEP_CANYON"


def build_fabric_table(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cell-native fabric table + terrain-aware aspect encoding and built mask.

    Drops the naive street-aggregated ``svf``/``svf_count`` from the feature
    set (kept out of the fabric vector on purpose, see module docstring).
    """
    cols = [c for c in FABRIC_COLS if c in grid.columns]
    out = grid[cols + ["geometry"]].copy()

    aspect_rad = np.deg2rad(out["aspect_deg"])
    out["northness"] = np.cos(aspect_rad)
    out["eastness"] = np.sin(aspect_rad)
    # Lenient variant ((λp>0.01)|(building_count>0)); identical to the canonical
    # building_count>0 mask on the campaign grids (pooled n=64,389).
    out["built_mask"] = built_mask(out, lenient=True)
    return out


def build_experience_table(
    svf_solar: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    openness: gpd.GeoDataFrame | None = None,
    hw: gpd.GeoDataFrame | None = None,
    hw_tol: float = 3.0,
) -> gpd.GeoDataFrame:
    """Per-observer experience table enriched with its cell's fabric covariates.

    ``openness`` is joined 1:1 when it shares the observer set (verified by
    length + coordinate match), otherwise by nearest neighbour. ``hw`` is a
    coarser, separate sampling and is attached by nearest within ``hw_tol`` m
    (flagged ``has_hw``); absent for calibration sites.
    """
    obs = svf_solar.copy().reset_index(drop=True)

    if openness is not None:
        obs["openness_class"] = _align_openness(obs, openness)

    # areal → point: which cell each observer falls in, then carry its fabric.
    fabric_cols = [c for c in (["zone_id"] + FABRIC_COVARIATES) if c in grid.columns]
    obs = gpd.sjoin(
        obs, grid[fabric_cols + ["geometry"]], how="left", predicate="within"
    ).drop(columns=["index_right"], errors="ignore")

    if hw is not None and len(hw):
        hw_near = gpd.sjoin_nearest(
            obs[["geometry"]], hw[["HW", "geometry"]],
            how="left", max_distance=hw_tol, distance_col="_hw_dist",
        )
        hw_near = hw_near[~hw_near.index.duplicated(keep="first")]
        obs["HW"] = hw_near["HW"].to_numpy()
        obs["has_hw"] = obs["HW"].notna().to_numpy()
    else:
        obs["HW"] = np.nan
        obs["has_hw"] = False

    return obs


def summarize_streets_to_cells(
    obs: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    sun_threshold_h: float = SUN_THRESHOLD_H,
) -> pd.DataFrame:
    """Support-aware per-cell summaries of street observations.

    Distributional (p50 + worst-decile p10 + fraction-below-threshold), never a
    bare mean. Observer spacing is uniform (~1.5 m), so a within-cell point
    summary already approximates length-weighting; ``n_street_obs`` is carried as
    the support. Cells with no observation are returned with NaN summaries and
    ``has_street_support = False`` — not interpolated.
    """
    if "zone_id" not in obs.columns:
        raise ValueError("obs must carry zone_id (run build_experience_table first)")

    g = obs.dropna(subset=["zone_id"]).groupby("zone_id")
    rows = pd.DataFrame({
        "n_street_obs": g.size(),
        "svf_p50": g["svf"].median(),
        "svf_p10": g["svf"].quantile(0.10),
        "solar_winter_p50": g["solar_hours_winter"].median(),
        "solar_winter_p10": g["solar_hours_winter"].quantile(0.10),
        "solar_winter_frac_below2h": g["solar_hours_winter"].apply(
            lambda s: float((s < sun_threshold_h).mean())
        ),
    })
    if "openness_class" in obs.columns:
        rows["frac_deep_canyon"] = g["openness_class"].apply(
            lambda s: float((s == DEEP_CANYON_CLASS).mean())
        )
    if "HW" in obs.columns and obs["HW"].notna().any():
        rows["hw_p50"] = g["HW"].median()

    # reindex to every cell so empty cells are explicit NULL + flag, not dropped
    out = rows.reindex(grid["zone_id"].to_numpy())
    out["has_street_support"] = out["n_street_obs"].notna() & (out["n_street_obs"] > 0)
    out["n_street_obs"] = out["n_street_obs"].fillna(0).astype(int)
    out.index.name = "zone_id"
    return out.reset_index()


def _align_openness(
    obs: gpd.GeoDataFrame, openness: gpd.GeoDataFrame
) -> np.ndarray:
    """Return openness_class aligned to obs — 1:1 if same observer set, else nearest."""
    same = len(obs) == len(openness) and bool(
        np.allclose(obs.geometry.x.to_numpy(), openness.geometry.x.to_numpy(), atol=1e-3)
        and np.allclose(obs.geometry.y.to_numpy(), openness.geometry.y.to_numpy(), atol=1e-3)
    )
    if same:
        return openness["openness_class"].to_numpy()
    near = gpd.sjoin_nearest(
        obs[["geometry"]], openness[["openness_class", "geometry"]], how="left"
    )
    near = near[~near.index.duplicated(keep="first")]
    return near["openness_class"].to_numpy()
