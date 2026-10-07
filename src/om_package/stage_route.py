"""OM2 compute stage "route": walks, wind regimes, route repair, the point
table with every point measure, and the ONE horizon march (GPU) that the
P-10 sun tables, the shade stage and walk_points reuse.

Writes its shipped tables into the stage work directory under logical
names (``<layout key>.<ext>``); the package stage lays them out, so a
layout rename never reruns the march.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import p10_p11, route_flags
from .buffers import BUFFER_RADII_M, compute_buffer_variables
from .formvars import compute_form_variables, compute_street_orientation_deg
from .io_utils import write_table
from .neighbourhoods import communities_crossed, join_communities
from .quality import write_quality_report
from .routes import compute_route_geometry_flag, densify_route, route_length_m
from .sun_envelope import ENVELOPE_SLOT_MIN, route_centroid_latlon
from .ventilation import compute_ventilation_proxies
from .walks import load_walks
from .wind_regimes import season_regimes

ROUTE_ID_COLUMNS = ("point_id", "route_id", "seq", "distance_along_m", "height_m", "geometry")


def compute_route(om: str, paths, radii=BUFFER_RADII_M, extras_fn=None, repair=None):
    """One route's joined point table (P-02/P-03/P-04/P-06 and, for OM2,
    the extras). ``extras_fn(points_gdf)`` returns (extra_columns keyed by
    point_id, p07 extras): it runs on the fully joined table so its columns
    are written and quality-checked with every other variable. ``repair`` =
    (classification, buildings) from route_flags.classify_route: every
    measure is then computed at the repaired position, and the table keeps
    the traced position in geometry, x and y.

    Returns (table, variable_cols, quality_extra, info)."""
    route_json = paths.route_json(om)
    points = densify_route(route_json)
    points["route_geometry_flag"] = compute_route_geometry_flag(points, paths).to_numpy()

    meas = points if repair is None else route_flags.repaired_points(points, repair[0])
    form = compute_form_variables(meas, paths)
    if repair is not None:
        form["street_orientation_deg"] = compute_street_orientation_deg(points)
    vent = compute_ventilation_proxies(meas, form["street_orientation_deg"].to_numpy(), paths)
    nbhd = join_communities(points, paths)

    joined = meas.merge(form, on="point_id").merge(vent, on="point_id").merge(nbhd, on="point_id")
    table = joined.merge(compute_buffer_variables(meas, paths, radii=radii), on="point_id")
    quality_extra = None
    if extras_fn is not None:
        extra_cols, quality_extra = extras_fn(table)
        table = table.merge(extra_cols, on="point_id", how="left")
    repair_facts = None
    if repair is not None:
        table, repair_facts = route_flags.apply_to_table(
            table, repair[0], repair[1], points, compute_form_variables(points, paths))
        table["geometry"] = points.geometry.to_numpy()

    variable_cols = [c for c in table.columns if c not in ROUTE_ID_COLUMNS]
    info = {
        "route_id": om,
        "repair_facts": repair_facts,
        "length_m": route_length_m(route_json),
        "n_points": len(points),
        "communities_crossed": communities_crossed(points, paths),
    }
    return table, variable_cols, quality_extra, info


def quality_summary(quality: dict, variable_cols: list[str]) -> dict:
    return {
        "n_points": quality["n_points"],
        "n_columns_checked": len(variable_cols),
        "route_geometry_flagged_points": quality.get("route_geometry_flagged_points"),
    }


def route_stage(work: Path, *, paths, matched_dir: Path, geometry_epoch: str, window_start: str, window_end: str,
                dose_slot_min: int, device: str, log=print) -> dict:
    """Everything OM2 needs from the route and the march. Files written to
    ``work``: route_points.{gpkg,parquet,csv}, quality_report.{json,csv},
    sun_envelope.{parquet,csv}, sun_dose.parquet, horizon_profiles.parquet."""
    t0 = time.time()

    def lap(label):
        nonlocal t0
        log(f"[build_om_package] time route/{label}: {time.time() - t0:.1f} s")
        t0 = time.time()

    log(f"[build_om_package] walks: reading {matched_dir} ...")
    walks_df, fixes = load_walks(matched_dir, paths.route_json("OM_2"))
    walk_dates = sorted(str(d) for d in walks_df["date"].unique())
    season = season_regimes(paths.root)
    regimes = p10_p11.campaign_regime_list(season)
    log(f"[build_om_package] {len(walks_df)} walks on {len(walk_dates)} dates; campaign regimes: "
        + ", ".join(f"{g['name']} {g['mean_direction_deg']:.1f} deg" for g in regimes))
    lap("walks + wind regimes")
    log("[build_om_package] route repair: classifying points, GPS consensus, medial axis ...")
    rep_res, rep_fixes, rep_buildings = route_flags.classify_route(densify_route(paths.route_json("OM_2")), paths, matched_dir)
    log("[build_om_package] route repair classes: "
        + str({c: int((rep_res['point_class'] == c).sum()) for c in route_flags.CLASSES}))
    lap("route repair classification")

    march: dict = {}

    def om2_extras(points_gdf):
        log(f"[build_om_package] P-10/P-11: horizon march on {device} ({len(points_gdf)} points) ...")
        lap("point measures before the march")
        horizon_deg, horizon_az, horizon_tab = p10_p11.horizon_arrays_and_table(points_gdf, paths, device=device)
        lap("horizon march")
        lat, lon = route_centroid_latlon(points_gdf)
        new_cols = p10_p11.new_point_columns(points_gdf, horizon_deg, horizon_az, horizon_tab,
                                             regimes=regimes, lat=lat, lon=lon)
        sun = p10_p11.sun_tables(horizon_tab, walk_dates, lat=lat, lon=lon, window_start=window_start,
                                 window_end=window_end, dose_slot_min=dose_slot_min)
        p10_summary = {**sun["summary"], "envelope_slot_min": ENVELOPE_SLOT_MIN}
        lap("P-10/P-11 point columns + sun tables")
        march.update(horizon_deg=horizon_deg, azimuths_deg=horizon_az, horizon_tab=horizon_tab, sun=sun,
                     p10_summary=p10_summary, lat=lat, lon=lon, point_ids=points_gdf["point_id"].to_numpy())
        regime_info = [{k: g[k] for k in ("key", "name", "slug", "mean_direction_deg")} for g in regimes]
        quality_extra = {"p10_p11": {
            "geometry_epoch": geometry_epoch,
            "p10": {k: p10_summary[k] for k in ("window", "tz", "n_days", "n_daylight_point_slots", "date_dependent_share",
                                              "class_share_of_daylight", "dose_slot_min")},
            "p11": {"campaign_regimes": regime_info,
                    "note": "ventilation columns are geometry-derived PROXIES; SBGL wind is an airport reference, not wind at the route"},
            "walks": {"n_walks": int(len(walks_df)), "n_walk_dates": len(walk_dates),
                      "n_partial": int(walks_df["partial"].sum())},
        }}
        return new_cols, quality_extra

    table, variable_cols, quality_extra, info = compute_route("OM_2", paths, extras_fn=om2_extras,
                                                              repair=(rep_res, rep_buildings))
    lap("route repair applied to the table")
    assert list(table["point_id"]) == list(march["point_ids"]), "points table order differs from the horizon march order"
    written = write_table(table, work, "route_points", geo=True)
    quality = write_quality_report(table, variable_cols, work, extra=quality_extra)

    sun = march["sun"]
    write_table(sun["envelope"], work, "sun_envelope")
    sun["dose"].to_parquet(work / "sun_dose.parquet", index=False)  # parquet only: the CSV was 234 MB
    march["horizon_tab"].to_parquet(work / "horizon_profiles.parquet", index=False)
    log(f"[build_om_package] P-10/P-11: envelope {len(sun['envelope'])} rows, dose {len(sun['dose'])} rows, "
        f"horizon {len(march['horizon_tab'])} rows")
    lap("writing route tables")

    repair_facts = info.pop("repair_facts")
    return {
        "route_result": {**info, "route_point_exts": [p.suffix.lstrip(".") for p in written],
                         "quality_summary": quality_summary(quality, variable_cols)},
        "repair_facts": repair_facts,
        "repair_res": rep_res,
        "repair_fixes": rep_fixes,
        "walks": walks_df,
        "walk_fixes": fixes,
        "walk_dates": walk_dates,
        "season": season,
        "regimes": regimes,
        "horizon_deg": march["horizon_deg"],
        "horizon_tab": march["horizon_tab"],
        "azimuths_deg": march["azimuths_deg"],
        "p10_summary": march["p10_summary"],
        "latlon": [march["lat"], march["lon"]],
    }
