"""P-06 — ventilation PROXIES per OM2 point.

Every column here is a PROXY for airflow, not a simulated or measured wind
field (this repo runs no flow simulation for P1 — see docs/p1_column_allowlist.json).
Four proxies, all airborne/geometry-derived:

  - ventilation_wind_alignment_proxy: how well the street's own axis lines
    up with the frequency-weighted prevailing wind bearing (data/maré/
    wind_rose.json, ASOS Galeão 2015-2024). 1.0 = street axis parallel to
    the prevailing wind (a channelling geometry), 0.0 = perpendicular (a
    blocking geometry). Undirected: a canyon channels wind from either end,
    so both the street axis and the wind bearing are folded to a 0-180 deg
    axis before comparing.
  - ventilation_frontal_area_proxy: nearest features_grid cell's
    lambda_f_mean, an OMNIDIRECTIONAL obstruction-density proxy (Oke 1988
    frontal-area density, averaged over 8 compass directions —
    src/urban_morphology.py). It says nothing about upwind fetch by
    itself — the per-direction columns below exist for that.
  - lambda_f_N, lambda_f_NE, lambda_f_E, lambda_f_SE, lambda_f_S,
    lambda_f_SW, lambda_f_W, lambda_f_NW: the same nearest features_grid
    cell's 8 per-compass-direction frontal-area densities, passed through
    unchanged so a windward-specific proxy can be built downstream (e.g.
    picking the column matching a given wind bearing) without this
    package guessing which direction matters for a given analysis.
  - ventilation_openness_proxy: nearest features_grid cell's porosity
    (1 - built volume / canopy volume in that 10 m cell — src/morphometry/
    indicators.py).
  - ventilation_dist_open_space_proxy_m: planar distance to the nearest
    features_grid cell classified "open" (lambda_p < OPEN_SPACE_LAMBDA_P_MAX)
    among 10 m grid cells with a valid lambda_p.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .io_utils import Paths
from .join_utils import nearest_join

MAX_JOIN_DIST_GRID_M = 12.0
#: A 10 m cell counts as "open space" below this plan density.
OPEN_SPACE_LAMBDA_P_MAX = 0.05

_COMPASS_BEARING_DEG = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
#: features_grid's 8 per-direction frontal-area columns, joined through
#: unchanged (must-fix 4, panel 2026-09-24).
LAMBDA_F_DIRECTION_COLS = [f"lambda_f_{d}" for d in _COMPASS_BEARING_DEG]


def prevailing_wind_bearing_deg(wind_rose_path) -> float:
    """Frequency-weighted circular mean bearing (deg, 0-360) of wind_rose.json."""
    d = json.loads(open(wind_rose_path).read())
    freqs = d["frequencies"]
    vx = vy = 0.0
    for direction, f in freqs.items():
        theta = math.radians(_COMPASS_BEARING_DEG[direction])
        vx += f * math.sin(theta)
        vy += f * math.cos(theta)
    bearing = math.degrees(math.atan2(vx, vy)) % 360.0
    return bearing


def wind_alignment_proxy(street_orientation_deg: np.ndarray, wind_bearing_deg: float) -> np.ndarray:
    axis = street_orientation_deg % 180.0
    wind_axis = wind_bearing_deg % 180.0
    diff = np.abs(axis - wind_axis)
    diff = np.minimum(diff, 180.0 - diff)  # fold to [0, 90]
    return np.cos(np.radians(diff))


def compute_ventilation_proxies(points_gdf, street_orientation_deg: np.ndarray, paths: Paths) -> pd.DataFrame:
    xy = np.column_stack([points_gdf.geometry.x.to_numpy(), points_gdf.geometry.y.to_numpy()])
    out = pd.DataFrame({"point_id": points_gdf["point_id"].to_numpy()})

    grid = pd.read_parquet(
        paths.features_grid,
        columns=["centroid_x", "centroid_y", "lambda_f_mean", "porosity", "lambda_p", *LAMBDA_F_DIRECTION_COLS],
    )
    grid_xy = grid[["centroid_x", "centroid_y"]].to_numpy()

    fa_cols = ["lambda_f_mean", *LAMBDA_F_DIRECTION_COLS]
    fa_join = nearest_join(xy, grid_xy, grid[fa_cols].reset_index(drop=True), MAX_JOIN_DIST_GRID_M)
    out["ventilation_frontal_area_proxy"] = fa_join["lambda_f_mean"]
    for col in LAMBDA_F_DIRECTION_COLS:
        out[col] = fa_join[col]

    op_join = nearest_join(xy, grid_xy, grid[["porosity"]].reset_index(drop=True), MAX_JOIN_DIST_GRID_M)
    out["ventilation_openness_proxy"] = op_join["porosity"]

    wind_bearing = prevailing_wind_bearing_deg(paths.wind_rose_json)
    out["ventilation_wind_alignment_proxy"] = wind_alignment_proxy(street_orientation_deg, wind_bearing)

    open_cells = grid[grid["lambda_p"].notna() & (grid["lambda_p"] < OPEN_SPACE_LAMBDA_P_MAX)]
    if len(open_cells) > 0:
        open_xy = open_cells[["centroid_x", "centroid_y"]].to_numpy()
        from scipy.spatial import cKDTree

        tree = cKDTree(open_xy)
        dist, _ = tree.query(xy, k=1)
        out["ventilation_dist_open_space_proxy_m"] = dist
    else:
        out["ventilation_dist_open_space_proxy_m"] = np.nan

    return out
