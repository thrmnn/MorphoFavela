"""Wire the route repair (route_repair.py) into the OM2 build.

Rules fixed by the project lead:
  projected        every measure is computed at the repaired position.
  beco             stays at its traced position (open ground). The GPS consensus
                   is evidence only: moves of 8 to 32 m went onto a noisy cloud.
  covered_passage  position and measures kept (real shade under a building).
  unresolved       position kept; street form measures that mean nothing inside a
                   footprint are left empty (UNRESOLVED_EMPTY).
Sky view factor is the build's own value (airborne grid, nearest valid cell) at
the repaired position, not route_repair.horizon_svf, which runs about 0.17 higher.
Height-to-width ratio uses the facade to facade width from building rays.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .route_repair import (CLASSES, COVERED_MIN_WALKS, GPS_NEAR_M, building_height_hw, classify_points, load_gps_fixes, street_width_m,
                           stretch_agreement, tangent_deg)
from .routes import UTM23S

UNRESOLVED_EMPTY = ["sky_view_factor", "height_width_ratio", "plan_density_lambda_p"]
POINT_COLUMNS = ["point_class", "x_repaired", "y_repaired", "shift_m", "street_width_m", "street_width_layer_m",
                 "street_width_capped", "building_height_layer_m"]


def classify_route(points: gpd.GeoDataFrame, paths, matched_dir) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """(result, fixes, buildings). Becos are put back on their traced position;
    the consensus and medial axis columns stay as evidence."""
    buildings = gpd.read_file(paths.buildings_mare)
    streets = gpd.read_file(paths.street_mare)
    fixes = load_gps_fixes(matched_dir)
    res = classify_points(points, buildings, streets, fixes)
    beco = (res["point_class"] == "beco").to_numpy()
    res["beco_gps_suggested_shift_m"] = np.where(beco, res["shift_m"], np.nan)
    res.loc[beco, "x_repaired"] = res.loc[beco, "x_original"]
    res.loc[beco, "y_repaired"] = res.loc[beco, "y_original"]
    res.loc[beco, "shift_m"] = 0.0
    res.loc[beco, "position_source"] = "route"
    return res, fixes, buildings


def repaired_points(points: gpd.GeoDataFrame, res: pd.DataFrame) -> gpd.GeoDataFrame:
    out = points.copy()
    out["geometry"] = gpd.points_from_xy(res["x_repaired"].to_numpy(), res["y_repaired"].to_numpy())
    return out.set_geometry("geometry", crs=UTM23S)


def apply_to_table(table: pd.DataFrame, res: pd.DataFrame, buildings, points: gpd.GeoDataFrame,
                   form_before: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add the repair columns to the joined points table (rows in route order)
    and return it with the before and after facts. form_before: the form
    variables computed at the traced positions."""
    t = table.copy()
    rep = repaired_points(points, res)
    sw = street_width_m(rep, buildings, tangent_bearing_deg=tangent_deg(points))
    hh = building_height_hw(sw)
    t["point_class"] = res["point_class"].to_numpy()
    t["x_repaired"] = res["x_repaired"].to_numpy()
    t["y_repaired"] = res["y_repaired"].to_numpy()
    t["shift_m"] = res["shift_m"].to_numpy()
    t["street_width_layer_m"] = t["street_width_m"]
    t["building_height_layer_m"] = t["building_height_m"]
    t["street_width_m"] = sw["street_width_m"].to_numpy()
    t["street_width_capped"] = sw["width_capped"].to_numpy() & sw["street_width_m"].notna().to_numpy()
    ray_h = hh["building_height_m"].to_numpy()
    t["building_height_m"] = np.where(np.isfinite(ray_h), ray_h, t["building_height_layer_m"])
    t["height_width_ratio"] = hh["height_width_ratio"].to_numpy()
    t["route_geometry_flag"] = (res["point_class"] != "street").to_numpy()
    unres = (res["point_class"] == "unresolved").to_numpy()
    for c in UNRESOLVED_EMPTY:
        t.loc[unres, c] = np.nan

    cls = res["point_class"].to_numpy()
    proj = cls == "projected"
    street = cls == "street"

    def med(s):
        s = pd.Series(np.asarray(s, float)).dropna()
        return float(s.median()) if len(s) else None

    gap = res["consensus_medial_gap_m"].dropna()
    both = street & t["street_width_m"].notna().to_numpy() & t["street_width_layer_m"].notna().to_numpy()
    d = (t["street_width_m"] - t["street_width_layer_m"]).to_numpy()
    facts = {
        "counts": {c: int((cls == c).sum()) for c in CLASSES},
        "beco_gps_suggested_moved": {
            "n": int((res["beco_gps_suggested_shift_m"] > 0.01).sum()),
            "median_m": med(res["beco_gps_suggested_shift_m"][res["beco_gps_suggested_shift_m"] > 0.01]),
            "max_m": float(res["beco_gps_suggested_shift_m"].max()),
        },
        "covered_min_walks": COVERED_MIN_WALKS,
        "gps_near_m": GPS_NEAR_M,
        "gps_medial_gap_median_m": med(gap),
        "gps_medial_share_within_5m": float((gap <= 5).mean()) if len(gap) else None,
        "n_with_gps_consensus": int(len(gap)),
        "projected": {
            "n": int(proj.sum()),
            "shift_median_m": med(res.loc[proj, "shift_m"]),
            "shift_max_m": float(res.loc[proj, "shift_m"].max()) if proj.any() else None,
            "sky_view_factor_before_median": med(form_before["sky_view_factor"].to_numpy()[proj]),
            "sky_view_factor_after_median": med(t["sky_view_factor"].to_numpy()[proj]),
            "height_width_before_median": med(form_before["height_width_ratio"].to_numpy()[proj]),
            "height_width_after_median": med(t["height_width_ratio"].to_numpy()[proj]),
            "street_width_before_median_m": med(form_before["street_width_m"].to_numpy()[proj]),
            "street_width_after_median_m": med(t["street_width_m"].to_numpy()[proj]),
            "building_height_before_median_m": med(form_before["building_height_m"].to_numpy()[proj]),
            "building_height_after_median_m": med(t["building_height_m"].to_numpy()[proj]),
        },
        "street_points_width": {
            "n": int(both.sum()),
            "layer_median_m": med(t["street_width_layer_m"].to_numpy()[both]),
            "facade_median_m": med(t["street_width_m"].to_numpy()[both]),
            "median_difference_m": med(d[both]),
            "height_width_layer_median": med(form_before["height_width_ratio"].to_numpy()[street]),
            "height_width_facade_median": med(t["height_width_ratio"].to_numpy()[street]),
            "capped_share": float(t["street_width_capped"].to_numpy()[street].mean()),
        },
        "all_points_height_width": {
            "layer_median": med(form_before["height_width_ratio"]),
            "facade_median": med(t["height_width_ratio"]),
            "layer_missing": int(form_before["height_width_ratio"].isna().sum()),
            "facade_missing": int(t["height_width_ratio"].isna().sum()),
        },
        "unresolved_emptied": {c: int(t.loc[unres, c].isna().sum()) for c in UNRESOLVED_EMPTY},
    }
    return t, facts
