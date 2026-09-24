"""P-04 — airborne street-form variables per OM2 point.

Building height, plan density, street width, height-to-width ratio, street
orientation and sky-view factor, all derived from the 2019 buildings + DTM
"airborne" source. Terrestrial SVF is PENDING (needs the team's 2026 OM2
terrestrial scan) and is not a column here — see the data dictionary.

Sources (all EPSG:31983):
  - building_height_m, street_width_m, height_width_ratio:
    outputs/maré/morphometrics/canyon/hw_streets.gpkg (H, W, HW), nearest-
    joined. hw_streets is a canyon cross-section sample along the street
    network (src/urban_morphology.py's projected-width canyon method):
    H/W either side of each sample point, built from buildings_mare
    'altura' + mare_dtm.
  - plan_density_lambda_p: outputs/maré/features/features_grid.parquet
    (10 m grid cell containing/nearest the point), lambda_p = building
    footprint area fraction per cell.
  - grid_cell_id: that same nearest features_grid cell's zone_id, so
    downstream models can cluster/group by the 10 m grid a point falls in.
  - sky_view_factor: outputs/maré/svf_v2/svf_streets.gpkg 'svf', ray-cast
    at 1.5 m pedestrian height against the buildings+DTM mesh (src/svf_v2).
  - street_orientation_deg: computed directly from OM2's own chained
    geometry (local tangent bearing at each point, central difference),
    reported 0-180 deg as an undirected street axis (a street has no
    "direction").

Join gaps: a point farther than MAX_JOIN_DIST_M from its nearest source
sample gets NaN for that variable (never a guessed value) — see P-07.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from .io_utils import Paths
from .join_utils import nearest_join

#: hw_streets / svf_streets samples are along actual street centrelines;
#: OM2 hugs streets/sidewalks, so a generous-but-bounded radius catches
#: real matches without joining across an unrelated street.
MAX_JOIN_DIST_HW_M = 20.0
MAX_JOIN_DIST_SVF_M = 15.0
#: features_grid cells are 10 m squares; half-diagonal ~7.1 m, pad for edges.
MAX_JOIN_DIST_GRID_M = 12.0


def compute_street_orientation_deg(points_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Local tangent bearing at each point (central difference along the
    route's own seq order), degrees, undirected axis in [0, 180)."""
    xy = np.column_stack([points_gdf.geometry.x.to_numpy(), points_gdf.geometry.y.to_numpy()])
    n = len(xy)
    lo = np.clip(np.arange(n) - 1, 0, n - 1)
    hi = np.clip(np.arange(n) + 1, 0, n - 1)
    dx = xy[hi, 0] - xy[lo, 0]
    dy = xy[hi, 1] - xy[lo, 1]
    angle = np.degrees(np.arctan2(dy, dx)) % 180.0
    return angle


def compute_form_variables(points_gdf: gpd.GeoDataFrame, paths: Paths) -> pd.DataFrame:
    xy = np.column_stack([points_gdf.geometry.x.to_numpy(), points_gdf.geometry.y.to_numpy()])
    out = pd.DataFrame({"point_id": points_gdf["point_id"].to_numpy()})

    hw = gpd.read_file(paths.hw_streets)
    hw_xy = np.column_stack([hw.geometry.x.to_numpy(), hw.geometry.y.to_numpy()])
    hw_join = nearest_join(xy, hw_xy, hw[["H", "W", "HW"]].reset_index(drop=True), MAX_JOIN_DIST_HW_M)
    out["building_height_m"] = hw_join["H"]
    out["street_width_m"] = hw_join["W"]
    out["height_width_ratio"] = hw_join["HW"]
    out["building_height_join_dist_m"] = hw_join["_join_dist_m"]

    svf = gpd.read_file(paths.svf_streets)
    svf_xy = np.column_stack([svf.geometry.x.to_numpy(), svf.geometry.y.to_numpy()])
    svf_join = nearest_join(xy, svf_xy, svf[["svf"]].reset_index(drop=True), MAX_JOIN_DIST_SVF_M)
    out["sky_view_factor"] = svf_join["svf"]
    out["sky_view_factor_join_dist_m"] = svf_join["_join_dist_m"]

    grid = pd.read_parquet(paths.features_grid, columns=["centroid_x", "centroid_y", "lambda_p", "zone_id"])
    grid_xy = grid[["centroid_x", "centroid_y"]].to_numpy()
    grid_join = nearest_join(xy, grid_xy, grid[["lambda_p", "zone_id"]].reset_index(drop=True), MAX_JOIN_DIST_GRID_M)
    out["plan_density_lambda_p"] = grid_join["lambda_p"]
    out["plan_density_join_dist_m"] = grid_join["_join_dist_m"]
    # grid_cell_id: features_grid.zone_id from the SAME join used for
    # plan_density_lambda_p, so models can cluster the 10 m-grid variables
    # (must-fix 4, panel 2026-09-24). Int cast: nearest_join leaves an
    # unjoined row as NaN (float); Int64 keeps that nullable.
    out["grid_cell_id"] = grid_join["zone_id"].astype("Int64")

    out["street_orientation_deg"] = compute_street_orientation_deg(points_gdf)

    return out
