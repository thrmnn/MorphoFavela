"""P-08 — data dictionary: one row per variable ID, ever. IDs are never
reused; a variable retired in a later version keeps its row (marked
retired), it doesn't vanish. v0.1 has no retired variables yet.
"""
from __future__ import annotations

from .buffers import BUFFER_RADII_M
from .routes import ROUTE_FLAG_MAX_STREET_DIST_M

# id -> {definition, unit, source, method, limits, status}
# status: "computed" (present as a real column in v0.1's tables) or
# "PENDING" (no column in v0.1; listed here so the dictionary is the single
# place that enumerates every variable this package will ever carry).
_BASE: dict[str, dict] = {
    "point_id": {
        "definition": "PROVISIONAL point identifier, e.g. OM2-000042 — minted on the OSM-inferred route, not the team's own walked route.",
        "unit": "-",
        "source": "Octopus route file (Google Drive, PI-owned) + this package's code",
        "method": "route_id + zero-padded integer metres from route start (deterministic; stable across rebuilds of the same OSM-inferred route file)",
        "limits": "PROVISIONAL: the ID string is stable, but the place it names may move once v0.2 rebuilds on the team's om_routes.gpkg — a published old->new point_id crosswalk will accompany that release. Not comparable across different route files for the same OM route if the source route JSON changes.",
        "status": "computed",
    },
    "route_geometry_flag": {
        "definition": "True where the OSM-inferred route is defective at this point: it falls inside a building footprint, or lands off the real street network.",
        "unit": "bool",
        "source": "data/maré/raw/buildings_mare.shp, data/maré/raw/street_mare.shp",
        "method": f"within(buildings_mare) OR distance-to-nearest(street_mare centreline) > {ROUTE_FLAG_MAX_STREET_DIST_M:.0f} m (src/om_package/routes.py compute_route_geometry_flag)",
        "limits": "Only catches route-inference defects visible against these two layers; a route that is on-street but still not the team's actual walked path is not caught. See the README's route_geometry_flag caveat for the measured share of plan_density_lambda_p == 1.0 points this explains.",
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
        "limits": "UPPER BOUND under canopy: the ray-cast mesh is buildings + bare-earth terrain only, no vegetation, so a tree-covered point's real sky view is <= this value, never more. Airborne (2019 buildings) only; NaN beyond 15 m of any SVF sample.",
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
        "limits": "10 m-cell resolution, not a point-native measurement; NaN beyond 12 m of any grid cell centroid. Some lambda_p == 1.0 points are a route_geometry_flag defect (route cuts through a building) rather than a real fully-built cell — see the README's route_geometry_flag caveat for the measured split.",
        "status": "computed",
    },
    "plan_density_join_dist_m": {
        "definition": "Distance from the OM2 point to the features_grid cell centroid used for plan_density_lambda_p.",
        "unit": "m", "source": "this package", "method": "nearest-neighbour KDTree distance", "limits": "-",
        "status": "computed",
    },
    "grid_cell_id": {
        "definition": "features_grid.zone_id of the same 10 m grid cell used for plan_density_lambda_p, so downstream models can cluster/group by grid cell (adjacent OM2 points are not independent).",
        "unit": "-", "source": "outputs/maré/features/features_grid.parquet column zone_id",
        "method": "same nearest-neighbour join (<=12 m) as plan_density_lambda_p — same source row, so the two columns are always consistent", "limits": "NaN (nullable Int64) beyond 12 m of any grid cell centroid, same gap as plan_density_lambda_p.",
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
        "definition": "PROXY for OMNIDIRECTIONAL obstruction density — Oke (1988) frontal-area density of the nearest 10 m grid cell, averaged over 8 compass directions. Not windward-specific by itself; see the lambda_f_<dir> columns for that.",
        "unit": "proxy, lambda_f (dimensionless)", "source": "outputs/maré/features/features_grid.parquet column lambda_f_mean",
        "method": "nearest-neighbour join (<=12 m); lambda_f_mean = mean over 8 compass directions, src/urban_morphology.py",
        "limits": "PROXY, not a simulated flow field; isotropic buffer, so it says nothing about upwind fetch on its own. 10 m-cell resolution.",
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

_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
for _d in _DIRECTIONS:
    _BASE[f"lambda_f_{_d}"] = {
        "definition": f"Frontal-area density (Oke 1988) of the nearest 10 m grid cell, facing {_d}.",
        "unit": "dimensionless", "source": f"outputs/maré/features/features_grid.parquet column lambda_f_{_d}",
        "method": "nearest-neighbour join (<=12 m), same join as ventilation_frontal_area_proxy; src/urban_morphology.py",
        "limits": "Geometry-derived, not a simulated flow field. 10 m-cell resolution; NaN beyond 12 m of any grid cell centroid.",
        "status": "computed",
    }
del _d

# P-03 segment columns (scripts/aggregate_om_points.py output — must-fix 4).
_BASE["segment_id"] = {
    "definition": "0-based segment index along the route, at the chosen segment length.",
    "unit": "-", "source": "src/om_package/segments.py aggregate_to_segments",
    "method": "distance_along_m // segment_length_m", "limits": "Segment length is a caller choice (scripts/aggregate_om_points.py), not fixed at build time — not comparable across two outputs built with different segment lengths.",
    "status": "computed",
}
_BASE["segment_start_m"] = {
    "definition": "distance_along_m of the first point in this segment.",
    "unit": "m", "source": "src/om_package/segments.py aggregate_to_segments", "method": "min(distance_along_m) per segment", "limits": "-",
    "status": "computed",
}
_BASE["segment_end_m"] = {
    "definition": "distance_along_m of the last point in this segment.",
    "unit": "m", "source": "src/om_package/segments.py aggregate_to_segments", "method": "max(distance_along_m) per segment", "limits": "-",
    "status": "computed",
}
_BASE["n_points"] = {
    "definition": "Count of 1 m points aggregated into this segment.",
    "unit": "count", "source": "src/om_package/segments.py aggregate_to_segments", "method": "group size", "limits": "The last segment of a route is typically shorter than segment_length_m, so it has fewer points.",
    "status": "computed",
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
    "tree_shade": {
        "definition": "Whether tree canopy shades each OM2 point. RESERVED column in the shade table schema (SHADE_TABLE_COLUMNS) — present but always null, so the table's shape will not change again once this is computed.",
        "unit": "bool", "source": "PENDING — no DSM/canopy layer for Maré on disk",
        "method": "PENDING", "limits": "Building-only shade (the 'shaded' column) releases once campaign dates are known; tree_shade stays null until a canopy/DSM layer exists.", "status": "PENDING",
    },
    "airborne_vs_terrestrial_comparison": {
        "definition": "Comparison of airborne vs. terrestrial form-variable estimates along OM2.",
        "unit": "-", "source": "PENDING — needs sky_view_factor_terrestrial first",
        "method": "PENDING", "limits": "-", "status": "PENDING",
    },
    "height_change_2024_2026": {
        "definition": "Change in building/canopy height between the 2024 airborne LiDAR and the 2026 OM2 terrestrial field campaign.",
        "unit": "m", "source": "PENDING — data location being confirmed by T. Hermann",
        "method": "PENDING", "limits": "Name kept as height_change_2024_2026 for now; will be revisited (e.g. renamed to height_change_2019_2026) once the 2024 airborne dataset's existence and location are confirmed.", "status": "PENDING",
    },
}

_SHADE_TABLE_ONLY = {
    "timestamp": {"definition": "Clock timestamp of a shade evaluation (5-min step).", "unit": "datetime, LABELLED UTC in v0.1.2 (a stated operating-rule choice, NOT a resolution of the still-UNRESOLVED campaign timezone — see src/om_package/shade.py module docstring; tz is a required, no-default parameter of every shade/join function)", "source": "src/om_package/shade.py", "method": "pd.date_range over the requested time window", "limits": "-", "status": "computed"},
    "date": {"definition": "Calendar date of a shade evaluation.", "unit": "date", "source": "src/om_package/shade.py", "method": "-", "limits": "-", "status": "computed"},
    "sun_altitude_deg": {"definition": "Apparent solar elevation at the evaluation timestamp.", "unit": "degrees", "source": "pvlib.solarposition.get_solarposition", "method": "-", "limits": "-", "status": "computed"},
    "sun_azimuth_deg": {"definition": "Solar azimuth (clockwise from north) at the evaluation timestamp.", "unit": "degrees", "source": "pvlib.solarposition.get_solarposition", "method": "-", "limits": "-", "status": "computed"},
    "shaded": {"definition": "Whether a building (not tree) shades the point at this timestamp.", "unit": "bool", "source": "src/om_package/shade.py is_shaded()", "method": "sun altitude vs. marched horizon angle at the sun's azimuth (point_horizon_profiles(), wired v0.1.2, max_dist_m=100m — see README Known limits)", "limits": "v0.1.2 covers only the pilot's 5 campaign dates (one CSV per device pulled 2026-09-25); more dates arrive as more CSVs are pulled. tz='UTC' labelling, not a resolved local time.", "status": "computed"},
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
