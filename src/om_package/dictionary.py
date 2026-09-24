"""P-08 — data dictionary: one row per variable ID, ever. IDs are never
reused; a variable retired in a later version keeps its row (marked
retired), it doesn't vanish. v0.1 has no retired variables yet.
"""
from __future__ import annotations

from .buffers import BUFFER_RADII_M

# id -> {definition, unit, source, method, limits, status}
# status: "computed" (present as a real column in v0.1's tables) or
# "PENDING" (no column in v0.1; listed here so the dictionary is the single
# place that enumerates every variable this package will ever carry).
_BASE: dict[str, dict] = {
    "point_id": {
        "definition": "Stable OM2 point identifier, e.g. OM2-000042.",
        "unit": "-",
        "source": "Octopus route file (Google Drive, PI-owned) + this package's code",
        "method": "route_id + zero-padded integer metres from route start (deterministic; stable across rebuilds of the same route file)",
        "limits": "Not comparable across different route files for the same OM route if the source route JSON changes.",
        "status": "computed",
    },
    "route_id": {
        "definition": "Which OM route this point belongs to (OM_1..OM_4).",
        "unit": "-", "source": "route file 'route_id' field", "method": "copied verbatim", "limits": "-",
        "status": "computed",
    },
    "seq": {
        "definition": "0-based point index along the route, in route order.",
        "unit": "-", "source": "this package", "method": "enumerate() over densified points", "limits": "-",
        "status": "computed",
    },
    "distance_along_m": {
        "definition": "Distance from the route's first point, along the chained centreline.",
        "unit": "m", "source": "this package", "method": "cumulative arc length in EPSG:31983 after chaining route edges by edge_order",
        "limits": "Route length computed in projected UTM; differs from the route file's own per-edge WGS84 length by ~0.3-0.4% (grid convergence).",
        "status": "computed",
    },
    "height_m": {
        "definition": "Observer height used for every point-level airborne variable (pedestrian height).",
        "unit": "m", "source": "src/svf_v2/sampling.py 'pedestrian_height' default",
        "method": "constant 1.5 m — matches the height this repo's own airborne SVF/street outputs were sampled at, so OM2 joins against them without a height mismatch",
        "limits": "Not a measured field height; a modelling convention.",
        "status": "computed",
    },
    "x": {
        "definition": "Point easting.",
        "unit": "m, EPSG:31983 (SIRGAS 2000 / UTM 23S)", "source": "this package", "method": "geometry.x, flattened for the non-geo (parquet/csv) table export", "limits": "-",
        "status": "computed",
    },
    "y": {
        "definition": "Point northing.",
        "unit": "m, EPSG:31983 (SIRGAS 2000 / UTM 23S)", "source": "this package", "method": "geometry.y, flattened for the non-geo (parquet/csv) table export", "limits": "-",
        "status": "computed",
    },
    "neighbourhood": {
        "definition": "Maré community/sub-neighbourhood the point falls within.",
        "unit": "-", "source": "data/maré/neighbourhoods.gpkg",
        "method": "point-in-polygon spatial join (predicate='within')", "limits": "Null for a point outside all mapped community polygons.",
        "status": "computed",
    },
    "building_height_m": {
        "definition": "Canyon building height flanking the street at this point (airborne).",
        "unit": "m", "source": "outputs/maré/morphometrics/canyon/hw_streets.gpkg column H",
        "method": "nearest-neighbour join (<=20 m) to the canyon cross-section sample; H comes from buildings_mare 'altura' + mare_dtm via src/urban_morphology.py's projected-width canyon method",
        "limits": "NaN beyond 20 m of any canyon sample (e.g. very short spur segments).",
        "status": "computed",
    },
    "building_height_join_dist_m": {
        "definition": "Distance from the OM2 point to the hw_streets sample used for building_height_m/street_width_m/height_width_ratio.",
        "unit": "m", "source": "this package", "method": "nearest-neighbour KDTree distance", "limits": "-",
        "status": "computed",
    },
    "street_width_m": {
        "definition": "Canyon street width at this point (building face to building face).",
        "unit": "m", "source": "outputs/maré/morphometrics/canyon/hw_streets.gpkg column W",
        "method": "nearest-neighbour join (<=20 m), src/urban_morphology.py projected-width canyon method",
        "limits": "NaN beyond 20 m of any canyon sample.",
        "status": "computed",
    },
    "height_width_ratio": {
        "definition": "Canyon aspect ratio H/W at this point.",
        "unit": "-", "source": "outputs/maré/morphometrics/canyon/hw_streets.gpkg column HW",
        "method": "nearest-neighbour join (<=20 m)", "limits": "NaN beyond 20 m of any canyon sample.",
        "status": "computed",
    },
    "sky_view_factor": {
        "definition": "Fraction of the sky hemisphere visible at this point (airborne).",
        "unit": "fraction [0,1]", "source": "outputs/maré/svf_v2/svf_streets.gpkg column svf",
        "method": "nearest-neighbour join (<=15 m); ray-cast at 1.5 m pedestrian height against a buildings+DTM mesh (src/svf_v2, 145-patch Tregenza sky)",
        "limits": "Airborne (2019 buildings) only; NaN beyond 15 m of any SVF sample.",
        "status": "computed",
    },
    "sky_view_factor_join_dist_m": {
        "definition": "Distance from the OM2 point to the svf_streets sample used for sky_view_factor.",
        "unit": "m", "source": "this package", "method": "nearest-neighbour KDTree distance", "limits": "-",
        "status": "computed",
    },
    "plan_density_lambda_p": {
        "definition": "Building footprint area fraction of the 10 m grid cell nearest this point.",
        "unit": "fraction [0,1]", "source": "outputs/maré/features/features_grid.parquet column lambda_p",
        "method": "nearest-neighbour join (<=12 m) to grid cell centroid; lambda_p from src/urban_morphology.py",
        "limits": "10 m-cell resolution, not a point-native measurement; NaN beyond 12 m of any grid cell centroid.",
        "status": "computed",
    },
    "plan_density_join_dist_m": {
        "definition": "Distance from the OM2 point to the features_grid cell centroid used for plan_density_lambda_p.",
        "unit": "m", "source": "this package", "method": "nearest-neighbour KDTree distance", "limits": "-",
        "status": "computed",
    },
    "street_orientation_deg": {
        "definition": "Local street/route axis bearing at this point.",
        "unit": "degrees, undirected axis [0,180)", "source": "OM2's own chained route geometry",
        "method": "central-difference tangent bearing between the neighbouring points, folded to [0,180) since a street axis has no direction",
        "limits": "Noisy at route kinks (rounded corners); 1 m point spacing smooths most sensor GPS noise already baked into the inferred route.",
        "status": "computed",
    },
    "ventilation_wind_alignment_proxy": {
        "definition": "PROXY for how well the street channels the prevailing wind (not a flow simulation).",
        "unit": "proxy score [0,1], 1=axis parallel to prevailing wind", "source": "street_orientation_deg + data/maré/wind_rose.json",
        "method": "cos(acute angle between the undirected street axis and the frequency-weighted circular-mean wind bearing)",
        "limits": "PROXY. Ignores building-scale channelling/blocking geometry beyond axis alignment; wind rose is a single station (ASOS Galeão) 8 km+ from Maré.",
        "status": "computed",
    },
    "ventilation_frontal_area_proxy": {
        "definition": "PROXY for windward obstruction — Oke (1988) frontal-area density of the nearest 10 m grid cell.",
        "unit": "proxy, lambda_f (dimensionless)", "source": "outputs/maré/features/features_grid.parquet column lambda_f_mean",
        "method": "nearest-neighbour join (<=12 m); lambda_f_mean = mean over 8 compass directions, src/urban_morphology.py",
        "limits": "PROXY, not a simulated flow field. 10 m-cell resolution.",
        "status": "computed",
    },
    "ventilation_openness_proxy": {
        "definition": "PROXY for canopy openness — volumetric porosity of the nearest 10 m grid cell.",
        "unit": "proxy, fraction [0,1]", "source": "outputs/maré/features/features_grid.parquet column porosity",
        "method": "nearest-neighbour join (<=12 m); porosity = 1 - built volume / canopy volume, src/morphometry/indicators.py",
        "limits": "PROXY, not a simulated flow field. 10 m-cell resolution.",
        "status": "computed",
    },
    "ventilation_dist_open_space_proxy_m": {
        "definition": "PROXY for proximity to open space — planar distance to the nearest low-density (lambda_p<0.05) 10 m grid cell.",
        "unit": "proxy, m", "source": "outputs/maré/features/features_grid.parquet (lambda_p threshold)",
        "method": "KDTree nearest distance to any grid cell centroid with lambda_p < 0.05",
        "limits": "PROXY. Threshold (0.05) is a modelling choice, not a measured open-space boundary; grid-cell resolution 10 m.",
        "status": "computed",
    },
}

_BUFFER_TEMPLATES = {
    "lambda_p_buffer_{r}m": {
        "definition": "Building footprint area fraction within a {r} m circular buffer around the point.",
        "unit": "fraction [0,1]", "source": "data/maré/buildings_extended_300m.gpkg",
        "method": "sum(building-buffer intersection area) / (pi * {r}^2)", "limits": "Airborne (2019 buildings) only.",
    },
    "building_count_buffer_{r}m": {
        "definition": "Count of buildings intersecting a {r} m circular buffer around the point.",
        "unit": "count", "source": "data/maré/buildings_extended_300m.gpkg",
        "method": "spatial-index intersects() query against the buffer polygon", "limits": "-",
    },
    "building_height_mean_buffer_{r}m": {
        "definition": "Unweighted mean building height ('altura') over buildings intersecting a {r} m circular buffer.",
        "unit": "m", "source": "data/maré/buildings_extended_300m.gpkg column altura",
        "method": "mean('altura') over buildings whose footprint intersects the buffer", "limits": "NaN where no building intersects.",
    },
}

_PENDING: dict[str, dict] = {
    "sky_view_factor_terrestrial": {
        "definition": "Terrestrial (ground-instrument) sky-view factor at each OM2 point.",
        "unit": "fraction [0,1]", "source": "PENDING — needs the team's 2026 OM2 terrestrial scan/photography campaign",
        "method": "PENDING", "limits": "Not computed in v0.1 (spec P-04: airborne only).", "status": "PENDING",
    },
    "building_shade_per_5min": {
        "definition": "Whether a building shades each OM2 point at each 5-minute timestamp on a campaign date.",
        "unit": "bool", "source": "PENDING — campaign dates not known yet (PI, 2026-09-23)",
        "method": "src/om_package/shade.py compute_shade() — implemented, not run; empty-schema table shipped in v0.1",
        "limits": "-", "status": "PENDING",
    },
    "tree_shade": {
        "definition": "Whether tree canopy shades each OM2 point.",
        "unit": "bool", "source": "PENDING — no DSM/canopy layer for Maré on disk",
        "method": "PENDING", "limits": "-", "status": "PENDING",
    },
    "airborne_vs_terrestrial_comparison": {
        "definition": "Comparison of airborne vs. terrestrial form-variable estimates along OM2.",
        "unit": "-", "source": "PENDING — needs sky_view_factor_terrestrial first",
        "method": "PENDING", "limits": "-", "status": "PENDING",
    },
    "height_change_2024_2026": {
        "definition": "Change in building/canopy height between the 2019 airborne source and the 2026 OM2 field campaign.",
        "unit": "m", "source": "PENDING — no 2024/2026 height re-survey on disk",
        "method": "PENDING", "limits": "-", "status": "PENDING",
    },
}

_SHADE_TABLE_ONLY = {
    "timestamp": {"definition": "Local clock timestamp of a shade evaluation (5-min step).", "unit": "datetime (America/Sao_Paulo)", "source": "src/om_package/shade.py", "method": "pd.date_range over the requested time window", "limits": "-", "status": "PENDING"},
    "date": {"definition": "Calendar date of a shade evaluation.", "unit": "date", "source": "src/om_package/shade.py", "method": "-", "limits": "-", "status": "PENDING"},
    "sun_altitude_deg": {"definition": "Apparent solar elevation at the evaluation timestamp.", "unit": "degrees", "source": "pvlib.solarposition.get_solarposition", "method": "-", "limits": "-", "status": "PENDING"},
    "sun_azimuth_deg": {"definition": "Solar azimuth (clockwise from north) at the evaluation timestamp.", "unit": "degrees", "source": "pvlib.solarposition.get_solarposition", "method": "-", "limits": "-", "status": "PENDING"},
    "shaded": {"definition": "Whether a building (not tree) shades the point at this timestamp.", "unit": "bool", "source": "src/om_package/shade.py is_shaded()", "method": "sun altitude vs. marched horizon angle at the sun's azimuth", "limits": "Needs point_horizon_profiles() (real, not run in v0.1) to be non-empty.", "status": "PENDING"},
}


def full_dictionary(radii=BUFFER_RADII_M) -> dict[str, dict]:
    d = dict(_BASE)
    for template_id, template in _BUFFER_TEMPLATES.items():
        for r in radii:
            col_id = template_id.format(r=r)
            row = {k: (v.format(r=r) if isinstance(v, str) else v) for k, v in template.items()}
            row["status"] = "computed"
            d[col_id] = row
    for k, v in _PENDING.items():
        d[k] = v
    for k, v in _SHADE_TABLE_ONLY.items():
        d.setdefault(k, v)
    return d


def dictionary_dataframe(radii=BUFFER_RADII_M):
    import pandas as pd

    d = full_dictionary(radii)
    rows = []
    for col_id, meta in d.items():
        row = {"id": col_id}
        row.update(meta)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    cols = ["id", "definition", "unit", "source", "method", "limits", "status"]
    return df[cols]
