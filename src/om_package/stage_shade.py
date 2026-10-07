"""OM2 compute stage "shade" (P-05): the nodata floor of the extended DTM
under the OM2 points and the building-shade table on the walk dates,
written to the stage work directory as building_shade.parquet."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .routes import UTM23S
from .shade import OM2_SHADE_MAX_DIST_M, SHADE_STEP_MIN, compute_shade_local, nodata_floor_m

LOCAL_TZ = "America/Sao_Paulo"


def shade_stage(work: Path, *, paths, route_points: pd.DataFrame, walk_dates: list[str], lat: float, lon: float,
                horizon_deg, azimuths_deg, log=print) -> dict:
    pts = gpd.GeoDataFrame(route_points[["point_id"]],
                           geometry=gpd.points_from_xy(route_points["x_repaired"], route_points["y_repaired"]), crs=UTM23S)
    log("[build_om_package] P-05: measuring the nodata floor (shade.nodata_floor_m) ...")
    floor = nodata_floor_m(pts, paths)
    log(f"[build_om_package] P-05: nodata floor min={floor['min']:.1f}m median={floor['median']:.1f}m "
        f"max={floor['max']:.1f}m (OM2_SHADE_MAX_DIST_M={OM2_SHADE_MAX_DIST_M:g}m)")
    assert OM2_SHADE_MAX_DIST_M <= floor["min"], (
        f"OM2_SHADE_MAX_DIST_M={OM2_SHADE_MAX_DIST_M} exceeds the measured nodata floor "
        f"minimum {floor['min']:.1f}m — the horizon march would hit nodata; revisit "
        "shade.py's OM2_SHADE_MAX_DIST_M before shipping"
    )
    log(f"[build_om_package] P-05: shade on {len(walk_dates)} walk dates ...")
    summary, _ = compute_shade_local(route_points["point_id"], walk_dates, SHADE_STEP_MIN, lat, lon, LOCAL_TZ,
                                     horizon_deg, azimuths_deg, work / "building_shade.parquet")
    log(f"[build_om_package] P-05: {summary['n_rows']} rows across {summary['n_dates']} walk dates "
        f"({summary['shade_fraction_daylight_pct']}% in building shade, daylight only, {LOCAL_TZ})")
    return {"shade_summary": summary, "nodata_floor": floor}
