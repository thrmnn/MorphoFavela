"""P-08, data dictionary: one row per variable ID, ever. IDs are never
reused; a variable retired in a later version keeps its row (marked
retired), it doesn't vanish. No variable has been retired yet.
"""
from __future__ import annotations

from .buffers import BUFFER_RADII_M
from .routes import ROUTE_FLAG_MAX_STREET_DIST_M
from .shade import SHADE_STEP_MIN
from .sensor_match import DEFAULT_TAUS_S
from .sun_envelope import ENVELOPE_SLOT_MIN
from .vent_indices import DEFAULT_BUFFER_M, MACDONALD_A, MACDONALD_BETA, MACDONALD_CD, VON_KARMAN

# id -> {definition, unit, source, method, limits, status}
_BASE: dict[str, dict] = {
    "point_id": {
        "definition": "Point identifier, for example OM2-000042: the route name and the distance in metres from the start of the route.",
        "unit": "-",
        "source": "route file of the Octopus walk dataset; this package's code",
        "method": "route_id + zero-padded integer metres from route start (deterministic: the same route file always gives the same identifiers)",
        "limits": "Identifiers are only comparable between packages built on the same route file.",
        "status": "computed",
    },
    "route_geometry_flag": {
        "definition": "True where the route trace does not follow a mapped street at this point: true for every point whose point_class is not street.",
        "unit": "bool",
        "source": "data/maré/raw/buildings_mare.shp, data/maré/raw/street_mare.shp",
        "method": f"within(buildings_mare) OR distance-to-nearest(street_mare centreline) > {ROUTE_FLAG_MAX_STREET_DIST_M:.0f} m (src/om_package/routes.py compute_route_geometry_flag)",
        "limits": "Only catches defects visible against these two layers. Use point_class to tell the cases apart.",
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
        "method": "constant 1.5 m, the height at which the airborne sky view and street layers were sampled, so the joins use one height",
        "limits": "Not a measured field height; a modelling convention.",
        "status": "computed",
    },
    "x": {
        "definition": "Point easting on the route trace (before any repair).",
        "unit": "m, EPSG:31983 (SIRGAS 2000 / UTM 23S)", "source": "this package", "method": "geometry.x, flattened for the non-geo (parquet/csv) table export", "limits": "-",
        "status": "computed",
    },
    "y": {
        "definition": "Point northing on the route trace (before any repair).",
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
        "definition": "Height of the buildings flanking the street at this point: the mean of the two facades hit by the width rays (one facade if the other side hit nothing).",
        "unit": "m", "source": "buildings_mare 'altura'; point at the repaired position (x_repaired, y_repaired)",
        "method": "src/om_package/route_repair.py street_width_m rays; where no ray hits a building (inside a footprint) the value of building_height_layer_m is kept",
        "limits": "2019 footprints and heights. The same value as the street layer where both exist is not guaranteed; building_height_layer_m keeps the street-layer value.",
        "status": "computed",
    },
    "building_height_join_dist_m": {
        "definition": "Distance from the OM2 point to the hw_streets sample used for building_height_m/street_width_m/height_width_ratio.",
        "unit": "m", "source": "this package", "method": "nearest-neighbour KDTree distance", "limits": "-",
        "status": "computed",
    },
    "street_width_m": {
        "definition": "Street width at this point, from building face to building face.",
        "unit": "m", "source": "buildings_mare footprints; point at the repaired position",
        "method": "two rays perpendicular to the route direction, left and right, to the nearest footprint edge, each capped at 40 m (src/om_package/route_repair.py street_width_m); the width is the sum of the two distances",
        "limits": "Empty inside a building footprint (covered passages and unresolved points). Where a side meets no building within 40 m the width is capped and street_width_capped is true. Measured from the footprints, so it exists in becos, where the street layer has none.",
        "status": "computed",
    },
    "height_width_ratio": {
        "definition": "Height-to-width ratio of the street at this point: building_height_m divided by street_width_m (face to face).",
        "unit": "-", "source": "buildings_mare footprints; point at the repaired position",
        "method": "building_height_m / street_width_m from the same two rays",
        "limits": "Empty where street_width_m is empty, and for unresolved points (a point inside a building has no street). The street layer's own ratio is no longer used.",
        "status": "computed",
    },
    "sky_view_factor": {
        "definition": "Fraction of the sky hemisphere visible at this point (airborne).",
        "unit": "fraction [0,1]", "source": "outputs/maré/svf_v2/svf_streets.gpkg column svf",
        "method": "nearest-neighbour join (<=15 m) at the repaired position (x_repaired, y_repaired); ray-cast at 1.5 m pedestrian height against a buildings+DTM mesh (src/svf_v2, 145-patch Tregenza sky)",
        "limits": "Computed from 2019 buildings and terrain only. NaN beyond 15 m of any sky view sample, and for unresolved points (inside a building).",
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
        "limits": "10 m-cell resolution, not a point-native measurement; NaN beyond 12 m of any grid cell centroid, and for unresolved points. Points with plan_density_lambda_p equal to 1.0 are mostly covered passages or unresolved points in the traced route.",
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
        "method": "same nearest-neighbour join (<=12 m) as plan_density_lambda_p, same source row, so the two columns are always consistent", "limits": "NaN (nullable Int64) beyond 12 m of any grid cell centroid, same gap as plan_density_lambda_p.",
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
        "limits": "PROXY. Ignores building-scale channelling/blocking geometry beyond axis alignment; wind rose is a single station (Galeão airport) 8 km+ from Maré.",
        "status": "computed",
    },
    "ventilation_frontal_area_proxy": {
        "definition": "PROXY for obstruction density from all directions: the frontal area density (Oke, 1988) of the nearest 10 m grid cell, averaged over 8 compass directions. For one direction see the lambda_f_<dir> columns.",
        "unit": "proxy, dimensionless", "source": "outputs/maré/features/features_grid.parquet column lambda_f_mean",
        "method": "nearest-neighbour join (<=12 m); lambda_f_mean = mean over 8 compass directions, src/urban_morphology.py",
        "limits": "PROXY, not a simulated flow field; isotropic buffer, so it says nothing about upwind fetch on its own. 10 m-cell resolution.",
        "status": "computed",
    },
    "ventilation_openness_proxy": {
        "definition": "PROXY for openness, the volumetric porosity of the nearest 10 m grid cell.",
        "unit": "proxy, fraction [0,1]", "source": "outputs/maré/features/features_grid.parquet column porosity",
        "method": "nearest-neighbour join (<=12 m); porosity = 1 - built volume / total volume up to the canopy height, src/morphometry/indicators.py",
        "limits": "PROXY, not a simulated flow field. 10 m-cell resolution.",
        "status": "computed",
    },
    "ventilation_dist_open_space_proxy_m": {
        "definition": "PROXY for proximity to open space: planar distance to the nearest 10 m grid cell with a plan area density below 0.05.",
        "unit": "proxy, m", "source": "outputs/maré/features/features_grid.parquet (lambda_p threshold)",
        "method": "KDTree nearest distance to any grid cell centroid with lambda_p < 0.05",
        "limits": "PROXY. Threshold (0.05) is a modelling choice, not a measured open-space boundary; grid-cell resolution 10 m.",
        "status": "computed",
    },
}

# --- P-10 (sun exposure) and P-11 (ventilation indices with
# time-matched wind). Every ventilation row is a PROXY from building
# geometry; the wind rows are Galeão airport observations, not wind at the
# route; no row is a measured air temperature or airflow. ---------------
_GEOM = "2019 building geometry (the geometry epoch is a build parameter)"
_SUN_PROXY = (
    "Geometry-derived (building and terrain horizon vs. sun position), not measured sunlight: "
    "it ignores cloud."
)
_PREVAILING = (
    "evaluated at the prevailing wind bearing (frequency-weighted circular mean of the 2015-2024 Galeão airport "
    "climatology, data/maré/wind_rose.json; the same bearing P-06 uses)"
)
_VENT_LIMITS = (
    "PROXY, not simulated or measured air movement. Galeão airport is a regional reference, not wind at the "
    "route; a circular mean of a spread wind rose is a summary bearing, not a mode."
)
_BASE.update({
    "annual_sun_hours": {
        "definition": "Hours per year with the sun above both the geometric horizon and the marched building/terrain horizon at this point (static, like sky_view_factor). Geometry-derived PROXY for direct-sun exposure, not measured sunlight.",
        "unit": "h per year",
        "source": f"{_GEOM}; marched horizon (p10_horizon_profiles.parquet); pvlib solar position",
        "method": "10-min steps over one calendar year in Rio local time (America/Sao_Paulo); step counted sunlit when sun altitude > 0 and > the horizon angle at the sun's azimuth (nearest marched azimuth); hours = sunlit steps x step length (src/om_package/sun_envelope.py annual_sun_hours)",
        "limits": _SUN_PROXY + " Horizon march is limited to the DTM's valid radius so very distant obstructions are not seen.",
        "status": "computed",
    },
    "windward_lambda_f_prevailing": {
        "definition": "PROXY: frontal area density facing the prevailing wind (Oke, 1988; nearest 10 m grid cell), " + _PREVAILING + ".",
        "unit": "dimensionless",
        "source": "lambda_f_<dir> columns of this table (outputs/maré/features/features_grid.parquet)",
        "method": "circular linear interpolation between the two nearest of the 8 compass-direction columns at the prevailing bearing (src/om_package/vent_indices.py windward_lambda_f)",
        "limits": _VENT_LIMITS + " 10 m-cell resolution; NaN where the lambda_f_<dir> columns are NaN.",
        "status": "computed",
    },
    "canyon_alignment_prevailing_deg": {
        "definition": "PROXY: angle between the street axis and the prevailing wind axis, folded to 0-90 deg (0 = along the street, channelling; 90 = across), " + _PREVAILING + ".",
        "unit": "degrees [0,90]",
        "source": "street_orientation_deg of this table + data/maré/wind_rose.json",
        "method": "both the street axis and the wind axis are undirected, so the absolute difference is folded mod 180 and then to 0-90 (src/om_package/vent_indices.py canyon_alignment_deg)",
        "limits": _VENT_LIMITS + " Axis alignment only; says nothing about building-scale blocking. Same information as ventilation_wind_alignment_proxy on a different scale.",
        "status": "computed",
    },
    "upwind_shelter_deg_prevailing": {
        "definition": "PROXY: horizon (obstruction) angle at the upwind azimuth, i.e. how high the surroundings rise toward the prevailing wind, " + _PREVAILING + ".",
        "unit": "degrees",
        "source": f"p10_horizon_profiles.parquet (marched horizon of {_GEOM})",
        "method": "horizon angle at the marched azimuth nearest the wind bearing (src/om_package/vent_indices.py upwind_shelter_deg)",
        "limits": _VENT_LIMITS + " Horizon march is limited to the DTM's valid radius, so it sees obstructions only within that distance.",
        "status": "computed",
    },
    "z0_macdonald_m": {
        "definition": "PROXY: roughness length by Macdonald et al. (1998), from the 50 m buffer's plan area density, mean building height and windward frontal area density, " + _PREVAILING + ".",
        "unit": "m",
        "source": f"lambda_p_buffer_{DEFAULT_BUFFER_M}m, building_height_mean_buffer_{DEFAULT_BUFFER_M}m, windward_lambda_f_prevailing of this table",
        "method": f"Macdonald, Griffiths & Hall (1998), Atmos. Environ. 32(11):1857-1864: roughness length / mean height = (1 - displacement height / mean height) x exp(-[0.5 beta (Cd/kappa^2) (1 - displacement height / mean height) x frontal area density]^-0.5), A={MACDONALD_A:g}, beta={MACDONALD_BETA:g}, Cd={MACDONALD_CD:g}, kappa={VON_KARMAN:g} (src/om_package/vent_indices.py, Macdonald function)",
        "limits": _VENT_LIMITS + " Staggered-array constants applied to an irregular favela fabric; NaN where the 50 m buffer has no building (mean height undefined).",
        "status": "computed",
    },
    "zd_macdonald_m": {
        "definition": "PROXY: displacement height by Macdonald et al. (1998), from the 50 m buffer's plan density and mean building height.",
        "unit": "m",
        "source": f"lambda_p_buffer_{DEFAULT_BUFFER_M}m and building_height_mean_buffer_{DEFAULT_BUFFER_M}m of this table",
        "method": f"displacement height / mean height = 1 + A^(-plan area density) x (plan area density - 1), A={MACDONALD_A:g} (src/om_package/vent_indices.py, Macdonald function)",
        "limits": _VENT_LIMITS + " Does not depend on wind direction. NaN where the 50 m buffer has no building.",
        "status": "computed",
    },
    "open_space_fraction": {
        "definition": "PROXY: share of the 50 m circular buffer not covered by building footprints (1 minus the plan area density).",
        "unit": "fraction [0,1]",
        "source": f"lambda_p_buffer_{DEFAULT_BUFFER_M}m of this table ({_GEOM})",
        "method": f"1 - lambda_p_buffer_{DEFAULT_BUFFER_M}m (src/om_package/vent_indices.py compute_indices)",
        "limits": "PROXY for ventilation openness, not measured or simulated air movement. Footprint area only: streets, courtyards and empty plots all count as open; no height information.",
        "status": "computed",
    },
    # ---- p10 tables
    "local_slot": {
        "definition": "Local Rio clock time of day (HH:MM, slot start) of a P-10 row. Rio local time is a fixed UTC-3 offset (no daylight saving since 2019).",
        "unit": "HH:MM, America/Sao_Paulo local time",
        "source": "src/om_package/sun_envelope.py",
        "method": f"slot grid over the 24 h day; envelope table at {ENVELOPE_SLOT_MIN} min, dose table on its own (coarser) grid stated in manifest.json p10.dose_slot_min",
        "limits": "Rio local time (the loggers record UTC): convert a device timestamp with tz_convert('America/Sao_Paulo') before taking its slot.",
        "status": "computed",
    },
    "class": {
        "definition": "Sun class of a point at a local time of day over every day of the analysis window, counting only days with the sun up: always_sunlit, always_shaded, date_dependent, or night (sun down on every day).",
        "unit": "category",
        "source": f"p10_horizon_profiles.parquet ({_GEOM}); pvlib solar position",
        "method": "sun altitude vs. marched horizon at the sun's azimuth for every day in the window (manifest.json p10.window); always_* if the state is identical on every sun-up day, date_dependent otherwise (src/om_package/sun_envelope.py sun_envelope)",
        "limits": _SUN_PROXY + " 'Shaded' means building/terrain horizon.",
        "status": "computed",
    },
    "sunlit_day_share": {
        "definition": "Share of the sun-up days in the window on which this point is sunlit at this local time of day (geometry-derived proxy).",
        "unit": "fraction [0,1]; null for night slots",
        "source": "p10_sun_envelope (this definition's table)",
        "method": "sunlit sun-up days / sun-up days at this slot",
        "limits": _SUN_PROXY,
        "status": "computed",
    },
    "n_days_sun_up": {
        "definition": "Number of days in the window on which the sun is above the horizon at this local time of day.",
        "unit": "days",
        "source": "pvlib solar position",
        "method": "count over the window of apparent sun altitude > 0 at the slot",
        "limits": "Depends only on the slot and the window, not on the point.",
        "status": "computed",
    },
    "scope": {
        "definition": "Which case a P-10 row describes: a walk date (YYYY-MM-DD, Rio local date) or envelope_min / envelope_median / envelope_max (statistic over every day of the window).",
        "unit": "category",
        "source": "src/om_package/p10_p11.py",
        "method": "walk dates are the unique Rio local dates of the walks in p02b_walks",
        "limits": "-",
        "status": "computed",
    },
    "dose_1h_wh_m2": {
        "definition": "Clear-sky direct-beam dose on the horizontal plane over the 1 h up to and including this local slot (same day), zero while the point is shaded by the building horizon. Geometry-derived UPPER BOUND, not measured radiation.",
        "unit": "Wh/m2 (rounded to 0.1)",
        "source": f"pvlib Ineichen clear-sky DNI x sin(sun altitude); p10_horizon_profiles.parquet ({_GEOM})",
        "method": "per-slot beam energy = DNI x sin(altitude) x slot length, zero when shaded or sun down; trailing sum over 1 h (src/om_package/sun_envelope.py direct_sun_dose). Shipped as parquet only (a CSV would be about 234 MB). Rows with scope = a date use that date; envelope_* rows are min/median/max over the window. Slots where the sun is down on every day and all doses are zero are omitted.",
        "limits": _SUN_PROXY + " Clear sky makes it an upper bound; diffuse and reflected radiation are not included; the trailing window is clipped at 00:00.",
        "status": "computed",
    },
    "dose_2h_wh_m2": {
        "definition": "As dose_1h_wh_m2, summed over the 2 h up to and including this local slot.",
        "unit": "Wh/m2 (rounded to 0.1)", "source": "see dose_1h_wh_m2", "method": "see dose_1h_wh_m2 (trailing 2 h)",
        "limits": _SUN_PROXY + " Clear-sky upper bound.", "status": "computed",
    },
    "dose_3h_wh_m2": {
        "definition": "As dose_1h_wh_m2, summed over the 3 h up to and including this local slot.",
        "unit": "Wh/m2 (rounded to 0.1)", "source": "see dose_1h_wh_m2", "method": "see dose_1h_wh_m2 (trailing 3 h)",
        "limits": _SUN_PROXY + " Clear-sky upper bound.", "status": "computed",
    },
    "azimuth_deg": {
        "definition": "Azimuth (clockwise from north) of a marched horizon direction.",
        "unit": "degrees",
        "source": "src/om_package/shade.py point_horizon_profiles (145-patch Tregenza sky; patches sharing an azimuth collapse to one row)",
        "method": "distinct azimuths of the Tregenza patch directions",
        "limits": "Azimuth grid is the sky discretisation's, not a free choice; the sun's azimuth is matched to the nearest.",
        "status": "computed",
    },
    "horizon_deg": {
        "definition": "Marched horizon angle above the horizontal at this point and azimuth: the highest building or terrain obstruction angle along that direction at 1.5 m observer height.",
        "unit": "degrees",
        "source": f"{_GEOM}; dtm_extended_300m.tif + buildings_extended_300m.gpkg via src/brisa_solar WP-02/WP-04 horizon engine",
        "method": "max over the Tregenza patches sharing the azimuth of the marched obstruction angle, march radius max_dist_m",
        "limits": "Geometry only; limited to the march radius; cell resolution of the obstruction surface (1 m resampled from the DTM's native resolution).",
        "status": "computed",
    },
    "agreement_share": {
        "definition": "Share of daylight point-slots (sun up under either clock reading) whose sun state (sunlit / shaded / night) is identical whether the device clock logged UTC or Rio local time.",
        "unit": "fraction [0,1]",
        "source": "p10_clock_agreement (this definition's table)",
        "method": "per campaign date, 5-min slots of the logged clock read as UTC (A) or as local time (B); state per point from the marched horizon; night counts as its own state (src/om_package/sun_envelope.py exact_date_agreement)",
        "limits": "Low agreement means the unresolved clock matters for exact-date shade; use the envelope (class, sunlit_day_share) where it does. Geometry-derived, not measured.",
        "status": "computed",
    },
    "n_daylight_point_slots": {
        "definition": "Number of point x 5-min slot cells with the sun up under either clock reading, the denominator of agreement_share.",
        "unit": "count", "source": "p10_clock_agreement", "method": "count of point-slots with sun altitude > 0 under reading A or B", "limits": "-",
        "status": "computed",
    },
    # ---- p11 observed wind table
    "valid_utc": {
        "definition": "Observation time of an Galeão airport weather report, UTC.",
        "unit": "ISO 8601, UTC", "source": "Iowa Environmental Mesonet ASOS archive, Galeão airport station (provenance.wind_source in manifest.json)",
        "method": "as reported", "limits": "Galeão airport is a regional reference at 10 m, not wind at the route.",
        "status": "computed",
    },
    "valid_local": {
        "definition": "valid_utc converted to Rio local time (fixed UTC-3).",
        "unit": "ISO 8601, local time (no offset)", "source": "valid_utc", "method": "tz conversion to America/Sao_Paulo",
        "limits": "Offset from the tz database, not typed.", "status": "computed",
    },
    "drct": {
        "definition": "Wind direction the wind blows FROM at Galeão airport, degrees clockwise from north; empty for calm or variable reports.",
        "unit": "degrees", "source": "Galeão airport reports (Iowa ASOS archive)", "method": "as reported",
        "limits": "Observed at the airport, not at the route. Reported on a coarse direction grid by the source.",
        "status": "computed",
    },
    "speed_ms": {
        "definition": "Wind speed at Galeão airport, 10 m.",
        "unit": "m/s", "source": "Galeão airport reports (Iowa ASOS archive), given in knots",
        "method": "knots x the knot-to-m/s factor used by scripts/build_wind_rose.py",
        "limits": "Observed at the airport, not at the route.",
        "status": "computed",
    },
    "calm": {
        "definition": "True when the reported speed is below the calm threshold used by the climatology (scripts/build_wind_rose.py CALM_MS, recorded in manifest.json provenance.wind_source).",
        "unit": "bool", "source": "speed_ms", "method": "speed_ms < calm threshold", "limits": "Calm reports carry no usable direction and are never matched to a campaign time.",
        "status": "computed",
    },
    "variable_direction": {
        "definition": "True when a non-calm report has no direction (variable wind).",
        "unit": "bool", "source": "drct, speed_ms", "method": "not calm and direction missing", "limits": "Never matched to a campaign time.",
        "status": "computed",
    },
    "used_if_device_clock_utc": {
        "definition": "Campaign date (YYYY-MM-DD) this observation would be matched to if the device clock logged UTC; empty when not used.",
        "unit": "date or empty", "source": "p05b_campaign_windows + this table",
        "method": "for each 5-min step of a campaign walk window, the nearest Galeão airport report with a usable direction within the match gap (manifest.json p11.max_gap_min) (src/om_package/wind_obs.py wind_at)",
        "limits": "The device clock reading is UNKNOWN; this column and used_if_device_clock_local are the two readings. Time-matched wind is Galeão airport, not at the route.",
        "status": "computed",
    },
    "used_if_device_clock_local": {
        "definition": "As used_if_device_clock_utc, if the device clock logged Rio local time (UTC-3).",
        "unit": "date or empty", "source": "p05b_campaign_windows + this table", "method": "as used_if_device_clock_utc, device time shifted to UTC by the fixed local offset",
        "limits": "See used_if_device_clock_utc.", "status": "computed",
    },
})

_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
for _d in _DIRECTIONS:
    _BASE[f"lambda_f_{_d}"] = {
        "definition": f"Frontal area density (Oke, 1988) of the nearest 10 m grid cell, facing {_d}.",
        "unit": "dimensionless", "source": f"outputs/maré/features/features_grid.parquet column lambda_f_{_d}",
        "method": "nearest-neighbour join (<=12 m), same join as ventilation_frontal_area_proxy; src/urban_morphology.py",
        "limits": "Geometry-derived, not a simulated flow field. 10 m-cell resolution; NaN beyond 12 m of any grid cell centroid.",
        "status": "computed",
    }
del _d

# P-03 segment columns (scripts/aggregate_om_points.py output, must-fix 4).
_BASE["segment_id"] = {
    "definition": "0-based segment index along the route, at the chosen segment length.",
    "unit": "-", "source": "src/om_package/segments.py aggregate_to_segments",
    "method": "distance_along_m // segment_length_m", "limits": "Segment length is a caller choice (scripts/aggregate_om_points.py), not fixed at build time, not comparable across two outputs built with different segment lengths.",
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
        "method": "sum(building-buffer intersection area) / (pi * {r}^2)", "limits": "2019 buildings only.",
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

_SHADE_TABLE_ONLY = {
    "timestamp_local": {"definition": f"Rio local time of a shade evaluation ({SHADE_STEP_MIN}-min step), with the -03:00 offset.", "unit": "datetime, America/Sao_Paulo (UTC-3)", "source": "src/om_package/shade.py", "method": "local 5-min grid of each walk date, sun above the horizon only", "limits": "-", "status": "computed"},
    "timestamp_utc": {"definition": "The same instant as timestamp_local, in UTC (the loggers record UTC).", "unit": "datetime, UTC", "source": "src/om_package/shade.py", "method": "timestamp_local converted to UTC", "limits": "-", "status": "computed"},
    "timestamp": {"definition": f"Clock timestamp of a shade evaluation ({SHADE_STEP_MIN}-min step).", "unit": "-", "source": "src/om_package/shade.py", "method": "pd.date_range over the requested time window", "limits": "-", "status": "computed"},
    "date": {"definition": "Rio local calendar date: of a shade evaluation (p05_building_shade) or of a walk (p02b_walks).", "unit": "date, YYYY-MM-DD", "source": "src/om_package/shade.py; src/om_package/walks.py", "method": "local date of the timestamp", "limits": "-", "status": "computed"},
    "sun_altitude_deg": {"definition": "Apparent solar elevation at the evaluation timestamp.", "unit": "degrees", "source": "pvlib.solarposition.get_solarposition", "method": "-", "limits": "-", "status": "computed"},
    "sun_azimuth_deg": {"definition": "Solar azimuth (clockwise from north) at the evaluation timestamp.", "unit": "degrees", "source": "pvlib.solarposition.get_solarposition", "method": "-", "limits": "-", "status": "computed"},
    "shaded": {"definition": "True when the point gets no direct sun at this timestamp: a building or the terrain blocks the sun, or the sun is below the horizon (night, sun_altitude_deg <= 0). Night rows are no direct sun, not building shade: take building-shade shares over rows with sun_altitude_deg > 0 only.", "unit": "bool", "source": "src/om_package/shade.py is_shaded()", "method": "sun altitude vs. marched horizon angle at the sun's azimuth (point_horizon_profiles(), march distance 100 m)", "limits": "Computed on the walk dates only, daylight steps only, Rio local time slots.", "status": "computed"},
}


RETIRED_IN = "v0.3.0"
#: id -> what replaced it. A retired id keeps its row (ids are never reused).
_RETIRED = {
    "windward_lambda_f_prevailing": "frontal_area_density_windward_<regime>",
    "canyon_alignment_prevailing_deg": "canyon_alignment_deg_<regime>",
    "upwind_shelter_deg_prevailing": "upwind_shelter_angle_deg_<regime>",
    "z0_macdonald_m": "z0_macdonald_m_<regime>",
    "ventilation_wind_alignment_proxy": "canyon_alignment_deg_<regime>",
    "timestamp": "timestamp_local and timestamp_utc",
    "agreement_share": "removed with the clock sensitivity analysis (loggers record UTC)",
    "n_daylight_point_slots": "removed with the clock sensitivity analysis (loggers record UTC)",
    "used_if_device_clock_utc": "removed with the clock sensitivity analysis (loggers record UTC)",
    "used_if_device_clock_local": "removed with the clock sensitivity analysis (loggers record UTC)",
    "valid_utc": "p11_wind_regimes and p11_regime_by_hour (p11_wind_observed.csv is no longer shipped)",
    "valid_local": "p11_wind_regimes and p11_regime_by_hour (p11_wind_observed.csv is no longer shipped)",
    "drct": "p11_wind_regimes and p11_regime_by_hour (p11_wind_observed.csv is no longer shipped)",
    "speed_ms": "mean_speed_ms (p11_wind_observed.csv is no longer shipped)",
    "calm": "p11_regime_by_hour (p11_wind_observed.csv is no longer shipped)",
    "variable_direction": "p11_wind_observed.csv is no longer shipped",
    "sky_view_factor_terrestrial": "nothing: not part of this package",
    "tree_shade": "nothing: not part of this package",
    "airborne_vs_terrestrial_comparison": "nothing: not part of this package",
    "height_change_2024_2026": "nothing: not part of this package",
}

_WIND = ("Galeão airport reports at 10 m, a regional reference and not wind at the route; regimes come from the peaks "
         "of the smoothed 16-sector rose (src/om_package/wind_regimes.py).")
_REGIME_LIMITS = (
    "PROXY, not simulated or measured air movement. " + _WIND + " The regime direction is the mean of the reports "
    "nearest that rose peak."
)


def _row(definition, unit, source, method, limits="-"):
    return {"definition": definition, "unit": unit, "source": source, "method": method, "limits": limits,
            "status": "computed"}


_V030 = {
    "walk_id": _row("Identifier of one logger walk of OM2, OM2_<date>_<period> (duration appended only if two walks share both).", "-",
                    "file names of the walk dataset (data/maré/octopus/prerelease_v020/matched/)", "src/om_package/walks.py _walk_ids",
                    "The walks were collected by residents of Maré and the dataset is cleaned and structured by Cassiano and Vincent (Octopus team); the date in the id is the UTC date of the file name."),
    "period": _row("In p02b_walks and the p13_temperature_pairing tables: part of the day of the walk (morning or evening). In p11_wind_regimes and p11_regime_by_hour: the wind record the row describes (campaign = the campaign-season window; climatology = 2015-2024).", "category",
                   "walk file names; src/om_package/wind_regimes.py", "as named", "-"),
    "start_local": _row("Rio local time of the first logged row of the walk, with the -03:00 offset.", "ISO 8601, America/Sao_Paulo", "walk file", "first timestamp converted from UTC (loggers record UTC)", "-"),
    "start_utc": _row("The same instant as start_local, in UTC.", "ISO 8601, UTC", "walk file", "first logged timestamp", "-"),
    "end_local": _row("Rio local time of the last logged row of the walk, with the -03:00 offset.", "ISO 8601, America/Sao_Paulo", "walk file", "last timestamp converted from UTC", "-"),
    "end_utc": _row("The same instant as end_local, in UTC.", "ISO 8601, UTC", "walk file", "last logged timestamp", "-"),
    "duration_min": _row("Minutes from the first to the last logged row of the walk.", "min", "walk file", "end - start", "-"),
    "coverage_share": _row("Share of the route length between the first and the last on-route fix of the walk.", "fraction [0,1]", "walk file; OM2 route", "(last - first distance along the route) / route length", "A walk below 0.9 is flagged partial."),
    "share_on_route": _row("Share of the walk's logged rows matched to edges of the OM2 route.", "fraction [0,1]", "walk file; OM2 route", "rows on route / all rows", "Side-street detours are dropped from arrival times, not projected onto the route."),
    "share_interpolated": _row("Share of the on-route rows whose position the map matcher interpolated.", "fraction [0,1]", "walk file", "match_status == interpolated", "-"),
    "max_gap_s": _row("Longest time between two consecutive on-route fixes.", "s", "walk file", "max diff of on-route timestamps", "-"),
    "partial": _row("True when the walk covers less than 0.9 of the route.", "bool", "p02b_walks", "coverage_share < 0.9", "-"),
    "wind_regime": _row("Wind regime of the walk: the regime name of the nearest Galeão airport report to the walk's mid time, 'calm' if that report is calm, 'none' if it is more than 60 min away or has no direction.", "category (regime name, calm, none)", _WIND, "nearest report to the walk's mid time, classified to the nearer campaign regime peak (src/om_package/wind_regimes.py tag_walks)", "Airport wind, not wind at the route."),
    "wind_report_time_local": _row("Rio local time of the Galeão airport report used to tag the walk.", "ISO 8601, America/Sao_Paulo", "Galeão airport reports", "as reported, converted from UTC", "-"),
    "wind_report_time_utc": _row("The same instant as wind_report_time_local, in UTC.", "ISO 8601, UTC", "Galeão airport reports", "as reported", "-"),
    "wind_direction_deg": _row("Direction the wind blows FROM in the Galeão airport report used to tag the walk; empty for calm or none.", "degrees clockwise from north", "Galeão airport reports", "as reported", "Reported on a coarse direction grid by the source."),
    "wind_speed_ms": _row("Wind speed in the Galeão airport report used to tag the walk.", "m/s", "Galeão airport reports", "knots x the knot-to-m/s factor of scripts/build_wind_rose.py", "-"),
    "wind_report_minutes_from_mid": _row("Minutes from the walk's mid time to that report (negative = report before).", "min", "Galeão airport reports; walk file", "report time - walk mid time", "-"),
    "t_arrival_local": _row("Rio local time at which the walk reached this point, with the -03:00 offset.", "ISO 8601, America/Sao_Paulo", "walk file; OM2 route", "time interpolated linearly against distance along the route between on-route fixes (src/om_package/walks.py arrival_times)", "Distance is a running maximum, so arrival times never go backwards; interpolated across gaps (see arrival_source)."),
    "t_arrival_utc": _row("The same instant as t_arrival_local, in UTC.", "ISO 8601, UTC", "walk file", "as t_arrival_local", "-"),
    "arrival_source": _row("How the arrival time was obtained: gps (between fixes less than 60 s apart) or gap_interpolated (bracketed by a longer gap).", "category", "walk file", "src/om_package/walks.py arrival_times", "Rows outside the walk (outside_walk) are not shipped."),
    "shaded_at_arrival": _row("True when the point is in building shade, or the sun is below the horizon, at the moment the walk reaches it. Matched columns <...>_tau<s>s carry it as 0/1.", "bool", "p10_horizon_profiles; pvlib solar position", "sun altitude at t_arrival vs the marched horizon at the sun's azimuth (src/om_package/shade.py is_shaded)", "Geometry-derived (buildings and terrain); ignores cloud."),
    "dose_1h_before_wh_m2": _row("Clear-sky direct-beam energy on a horizontal plane in the 1 h before the walk reached this point; zero while the point is shaded by the building horizon.", "Wh/m2 (rounded to 0.1)", "pvlib Ineichen clear-sky DNI; p10_horizon_profiles", "integral over [t_arrival - 1 h, t_arrival] on a 1-min grid (src/om_package/walk_dose.py)", "Clear-sky upper bound, geometry-derived, not measured radiation."),
    "dose_3h_before_wh_m2": _row("As dose_1h_before_wh_m2, over the 3 h before arrival.", "Wh/m2 (rounded to 0.1)", "see dose_1h_before_wh_m2", "see dose_1h_before_wh_m2", "Clear-sky upper bound."),
    "regime_key": _row("Key of a wind regime: reg1 is the larger regime of the campaign season; climatology regimes take the key of the nearest campaign regime. 'calm' in p11_regime_by_hour.", "category", "src/om_package/wind_regimes.py", "see wind_regimes.find_regimes, assign_keys", "-"),
    "name": _row("Regime name: the 16-point compass name of the regime's mean direction (east-southeast, north-northwest ...).", "category", "src/om_package/wind_regimes.py", "compass_name(mean_direction_deg)", "Names follow the data; the climatology's name can differ from the campaign's."),
    "column_slug": _row("The regime name lowercased with hyphens and spaces as underscores; the suffix of the regime's point columns.", "category", "name", "src/om_package/p10_p11.py regime_slug", "-"),
    "mean_direction_deg": _row("Circular mean direction the wind blows FROM, of the reports assigned to the regime.", "degrees clockwise from north", _WIND, "circular mean of the member reports", "Airport wind, not wind at the route."),
    "share": _row("In p11_wind_regimes: share of the directional (non-calm) reports in the regime. In p11_regime_by_hour: share of the reports in that local hour in the regime or calm.", "fraction [0,1]", _WIND, "count / count", "Variable and missing-direction reports are excluded."),
    "mean_speed_ms": _row("Mean wind speed of the reports in the regime.", "m/s", "Galeão airport reports (Iowa ASOS archive), given in knots", "mean of the member reports", "Airport wind, 10 m."),
    "n_reports": _row("Number of reports in the regime.", "count", "Galeão airport reports", "count", "-"),
    "period_calm_share": _row("Share of all reports in the period that are calm, a bookkeeping column repeated on both regime rows.", "fraction [0,1]", "Galeão airport reports", "calm reports / all reports", "-"),
    "mixture_component_direction_deg": _row("Mean direction of the von Mises mixture component nearest the regime (the check on the regime split).", "degrees", "Galeão airport reports", "two von Mises components plus a uniform background fitted by EM to the reports with a seeded +/-5 degree jitter that undoes the 10 degree reporting steps (wind_regimes.vonmises_mixture)", "A check only: the regimes themselves come from the rose peaks."),
    "mixture_difference_deg": _row("Circular difference between the regime's mean direction and that mixture component.", "degrees", "Galeão airport reports", "see mixture_component_direction_deg", "-"),
    "mixture_component_weight": _row("Mixture weight of that component.", "fraction [0,1]", "Galeão airport reports", "see mixture_component_direction_deg", "-"),
    "mixture_component_kappa": _row("Concentration of that component (larger = narrower).", "-", "Galeão airport reports", "see mixture_component_direction_deg", "-"),
    "mixture_background_weight": _row("Weight of the uniform background in the mixture.", "fraction [0,1]", "Galeão airport reports", "see mixture_component_direction_deg", "-"),
    "local_hour": _row("Rio local hour of day (0-23) of the Galeão airport reports.", "hour", "Galeão airport reports", "valid time converted to America/Sao_Paulo", "-"),
    "regime": _row("In p11_regime_by_hour: the regime name or 'calm'.", "category", "src/om_package/wind_regimes.py hourly_frequency", "each report goes to the nearer regime peak; calm = speed below the calm threshold", "-"),
    "zd_macdonald_m": None,  # placeholder replaced below
}
del _V030["zd_macdonald_m"]

for _id in ("sky_view_factor_terrestrial", "tree_shade", "airborne_vs_terrestrial_comparison", "height_change_2024_2026"):
    _V030[_id] = _row("Identifier kept so that identifiers are never reused. No table has a column of this name.", "-", "-", "-", "-")

_MEASURE_NOTES = {
    "sky_view_factor": "sky view factor", "height_width_ratio": "height-to-width ratio",
    "building_height_m": "building height", "plan_density_lambda_p": "plan area density",
    "shaded_at_arrival": "shaded at arrival (0/1)", "dose_1h_before_wh_m2": "direct dose in the 1 h before arrival",
}


def _regime_rows(regimes: list[dict]) -> dict[str, dict]:
    out = {}
    for g in regimes:
        sl, nm = g["slug"], g["name"]
        where = f"at the mean direction of the campaign-season '{nm}' wind regime (p11_wind_regimes)"
        out[f"frontal_area_density_windward_{sl}"] = _row(
            f"PROXY: frontal area density (Oke, 1988; nearest 10 m grid cell) facing the wind, {where}.", "dimensionless",
            "lambda_f_<dir> columns of this table", "circular linear interpolation between the two nearest of the 8 compass-direction columns (src/om_package/vent_indices.py windward_lambda_f)",
            _REGIME_LIMITS + " 10 m-cell resolution; NaN where the lambda_f_<dir> columns are NaN.")
        out[f"canyon_alignment_deg_{sl}"] = _row(
            f"PROXY: angle between the street axis and the wind axis, folded to 0-90 deg (0 = along the street, channelling; 90 = across), {where}.", "degrees [0,90]",
            "street_orientation_deg of this table", "absolute difference of the undirected axes folded mod 180 and then to 0-90 (src/om_package/vent_indices.py canyon_alignment_deg)",
            _REGIME_LIMITS + " Axis alignment only; says nothing about building-scale blocking.")
        out[f"upwind_shelter_angle_deg_{sl}"] = _row(
            f"PROXY: horizon (obstruction) angle at the upwind azimuth, how high the surroundings rise toward the wind, {where}.", "degrees",
            f"p10_horizon_profiles.parquet (marched horizon of {_GEOM})", "horizon angle at the marched azimuth nearest the regime direction (src/om_package/vent_indices.py upwind_shelter_deg)",
            _REGIME_LIMITS + " Horizon march limited to the DTM's valid radius.")
        out[f"z0_macdonald_m_{sl}"] = _row(
            f"PROXY: roughness length by Macdonald et al. (1998) from the 50 m buffer's plan area density, mean building height and windward frontal area density, {where}.", "m",
            f"lambda_p_buffer_{DEFAULT_BUFFER_M}m, building_height_mean_buffer_{DEFAULT_BUFFER_M}m, frontal_area_density_windward_{sl} of this table",
            f"Macdonald, Griffiths & Hall (1998), Atmos. Environ. 32(11):1857-1864; A={MACDONALD_A:g}, beta={MACDONALD_BETA:g}, Cd={MACDONALD_CD:g}, kappa={VON_KARMAN:g} (src/om_package/vent_indices.py, Macdonald function)",
            _REGIME_LIMITS + " Staggered-array constants applied to an irregular favela fabric; NaN where the 50 m buffer has no building.")
    return out


def _matched_rows(regimes: list[dict], taus=DEFAULT_TAUS_S) -> dict[str, dict]:
    measures = {**_MEASURE_NOTES}
    for g in regimes:
        for stem in ("frontal_area_density_windward", "canyon_alignment_deg", "upwind_shelter_angle_deg"):
            measures[f"{stem}_{g['slug']}"] = f"{stem.replace('_', ' ')} at the {g['name']} regime"
    out = {}
    for m, note in measures.items():
        for tau in taus:
            out[f"{m}_tau{float(tau):g}s"] = _row(
                f"Sensor-matched {note}: the exponentially weighted mean of {m} over the points the walk had already passed, as seen by a sensor with time constant {float(tau):g} s.",
                "as " + m, "this table; walk file", f"sum w x / sum w over points j reached no later than this one and at most 5 tau earlier, w = exp(-dt / {float(tau):g} s) (src/om_package/sensor_match.py)",
                "tau is the 63 % response time (tau = t90 / ln 10). Empty where no point within 5 tau has a value. One value per walk and point; segment with aggregate_to_segments.py --by walk_id.")
    return out

_V031 = {
    "point_class": {
        "definition": "What the street map says about the point: street (on a mapped street), projected (inside a building outline by at most 4 m, moved to the nearest open ground), beco (open ground more than 10 m from a mapped street: an alley missing from the map), covered_passage (deeper inside a building outline, with GPS fixes from at least 10 walks within 8 m: a passage under a building) or unresolved (deeper inside a building outline with no GPS support).",
        "unit": "category", "source": "buildings_mare, street_mare, GPS fixes of the walks",
        "method": "src/om_package/route_repair.py classify_points, applied in the order street, projected, beco, covered_passage, unresolved; the build keeps beco, covered_passage and unresolved points at their traced position",
        "limits": "Footprints are 2019 and the route trace has its own offset. route_geometry_flag is true for every class except street.",
        "status": "computed",
    },
    "x_repaired": {
        "definition": "Easting of the position at which the measures of this point are computed.",
        "unit": "m, EPSG:31983 (SIRGAS 2000 / UTM 23S)", "source": "this package",
        "method": "projected points: nearest open ground, set back 0.5 m from the wall; all other points: equal to x",
        "limits": "Open ground is the space outside every footprint grown by 0.5 m.",
        "status": "computed",
    },
    "y_repaired": {
        "definition": "Northing of the position at which the measures of this point are computed.",
        "unit": "m, EPSG:31983 (SIRGAS 2000 / UTM 23S)", "source": "this package",
        "method": "as x_repaired", "limits": "-",
        "status": "computed",
    },
    "shift_m": {
        "definition": "Distance between the traced position (x, y) and the repaired position (x_repaired, y_repaired).",
        "unit": "m", "source": "this package", "method": "Euclidean distance", "limits": "Zero except for projected points.",
        "status": "computed",
    },
    "street_width_layer_m": {
        "definition": "Street width from the street layer's canyon sample (building face to building face), kept for comparison with street_width_m.",
        "unit": "m", "source": "outputs/maré/morphometrics/canyon/hw_streets.gpkg column W",
        "method": "nearest-neighbour join (<=20 m) at the repaired position",
        "limits": "NaN beyond 20 m of any canyon sample, which includes most becos.",
        "status": "computed",
    },
    "building_height_layer_m": {
        "definition": "Building height from the street layer's canyon sample, kept for comparison with building_height_m.",
        "unit": "m", "source": "outputs/maré/morphometrics/canyon/hw_streets.gpkg column H",
        "method": "nearest-neighbour join (<=20 m) at the repaired position", "limits": "NaN beyond 20 m of any canyon sample.",
        "status": "computed",
    },
    "street_width_capped": {
        "definition": "True where a width ray met no building within 40 m on one side, so street_width_m is a lower bound of the open width.",
        "unit": "bool", "source": "this package", "method": "ray length reached the 40 m cap on either side", "limits": "-",
        "status": "computed",
    },
}


_TP = "p13 temperature pairing (src/om_package/temp_pairing.py)"
_WALKT = "walk files (Temperature column)"
_LOGT = "outdoor fixed loggers of the Octopus team (src/om_package/fixed_loggers.py)"
_BOOT = "95% interval from resampling walks with replacement"


def p13_rows() -> dict[str, dict]:
    """Columns of the p13_temperature_pairing_* tables (first look at the walk temperature readings)."""
    r = _row
    return {
        # readings
        "t_utc": r("Time of the walk temperature reading, in universal time.", "ISO 8601, UTC", _WALKT, "timestamp_utc of the matched GPS fix", "-"),
        "t_local": r("The same instant as t_utc, in Rio local time with the -03:00 offset.", "ISO 8601, America/Sao_Paulo", _WALKT, "t_utc converted", "-"),
        "minutes_since_start": r("Minutes from the walk's first logged row to the reading.", "min", _WALKT, "t_utc - start_utc of p02b_walks", "Use it to leave out any start window, for example the first 15 evening minutes."),
        "temperature_c": r("Air temperature recorded by the walk sensor at the reading.", "°C", _WALKT, "as recorded; one reading every 5 s", "Sensor response time unknown; see the report's time constant section."),
        "background_c": r("Background temperature of the outdoor fixed loggers in Maré at the time of the reading.", "°C", _LOGT, "offset-corrected median of the outdoor loggers per minute, read on the Rio local clock, interpolated linearly; empty when no logger minute lies within 10 min", "Common outdoor level, not any one logger's absolute scale."),
        "anomaly_c": r("Reading minus logger background minus the walk's mean of that difference; for walks without logger cover, the reading minus a straight time trend fitted per walk.", "°C", _TP, "see anomaly_source", "Within-walk quantity: walk means are zero by construction."),
        "anomaly_source": r("How anomaly_c was formed: logger_background or walk_detrend (walk not fully covered by the loggers).", "category", _TP, "full cover = a logger minute within every minute of the walk", "-"),
        "anomaly_detrend_c": r("Reading minus a straight time trend fitted per walk, for every walk (comparison with anomaly_c).", "°C", _TP, "least squares on time, start minute left out of the fit", "Every walk runs from 0 m to the end, so this also removes any along-route gradient."),
        # tau scan
        "tau_s": r("Time constant of the sensor-matched measures (0 = the 1 m value, no smoothing).", "s", _TP, "sensor_match.sensor_matched", "-"),
        "r2_within": r("Share of the within-walk variation of anomaly_c explained by the model, in the fitted walks.", "fraction", _TP, "least squares after removing each walk's mean", "In-sample; see cv_r2."),
        "cv_r2": r("Share of the within-walk variation of anomaly_c predicted in walks left out of the fit, one walk at a time.", "fraction (negative = worse than the walk mean)", _TP, "leave one walk out", "-"),
        "r2_lo": r("Lower end of the interval of r2_within.", "fraction", _TP, _BOOT, "-"),
        "r2_hi": r("Upper end of the interval of r2_within.", "fraction", _TP, _BOOT, "-"),
        "share_boot_best": r("Share of walk resamples in which this time constant explains the most variation.", "fraction", _TP, "walk bootstrap", "-"),
        # events
        "event_distance_m": r("Distance along the route of the first point after a sun and shade change.", "m", _TP, "shaded_at_arrival changes, stable at least 30 m on each side, GPS arrival times, walking pace", "-"),
        "event_t_utc": r("Time the walk reached event_distance_m.", "ISO 8601, UTC", "p12_walk_points", "t_arrival_utc", "-"),
        "event_direction": r("sun_to_shade or shade_to_sun.", "category", _TP, "-", "-"),
        "before_m": r("Length of the stable stretch before the change.", "m", _TP, "-", "-"),
        "after_m": r("Length of the stable stretch after the change.", "m", _TP, "-", "-"),
        "before_s": r("Walking time along the stable stretch before the change.", "s", _TP, "-", "-"),
        "after_s": r("Walking time along the stable stretch after the change.", "s", _TP, "-", "-"),
        "used": r("True when the event has readings on both sides and enters the mean response.", "bool", _TP, "at least 2 readings before and 4 after", "-"),
        "bin_s": r("Centre of a 5 s bin of time from the change.", "s", _TP, "-", "-"),
        "mean_change_c": r("Mean change of anomaly_c from its level before the change, signed so a change into shade should read negative.", "°C", _TP, "per event bin means, then mean over events", "Later bins hold only the longest stretches."),
        "n_events": r("Number of events contributing to the bin.", "count", _TP, "-", "-"),
        # coefficients
        "measure": r("Street measure of the model term: shade, dose (1 h sun dose), svf (sky view factor) or hw (height-to-width ratio).", "category", _TP, "-", "-"),
        "matched_column": r("Sensor-matched column used for the term.", "-", _TP, "-", "-"),
        "per_unit": r("Change of the measure the effect refers to: shade 1 (fully sunlit to fully shaded recent path), dose 100 Wh/m2, svf 0.1, hw 1.", "unit of the measure", _TP, "-", "-"),
        "effect_c": r("Change of anomaly_c that goes with per_unit of the measure, others held fixed.", "°C", _TP, "least squares with walk fixed effects", "Association, not cause. Shade and dose are nearly collinear."),
        "effect_lo_c": r("Lower end of the 95% interval of effect_c.", "°C", _TP, "standard errors clustered by walk", "Ignores spatial correlation between walks, so likely too narrow."),
        "effect_hi_c": r("Upper end of the 95% interval of effect_c.", "°C", _TP, "standard errors clustered by walk", "As effect_lo_c."),
        "model": r("Model name: main (shade, dose, svf), shade_svf, with_ratio (adds hw), main_scan_tau (at the scan's best tau) or a sensitivity run with the evening start window cut.", "category", _TP, "-", "-"),
        # segment profile
        "segment": r("Index of the 20 m segment along the route (0 = first 20 m).", "-", _TP, "floor(distance_along_m / 20)", "-"),
        "segment_mid_m": r("Distance along the route of the segment's middle.", "m", _TP, "-", "-"),
        "mean_anomaly_logger_c": r("Mean of the per-walk segment means of the logger-background anomaly, walks with logger cover.", "°C", _TP, "-", "-"),
        "mean_anomaly_logger_lo_c": r("Lower end of the interval of mean_anomaly_logger_c.", "°C", _TP, _BOOT, "-"),
        "mean_anomaly_logger_hi_c": r("Upper end of the interval of mean_anomaly_logger_c.", "°C", _TP, _BOOT, "-"),
        "mean_anomaly_detrend_c": r("As mean_anomaly_logger_c, with the per-walk time trend removed instead.", "°C", _TP, "-", "-"),
        "mean_anomaly_start_c": r("As mean_anomaly_logger_c, for readings in the start window left out of the analysis.", "°C", _TP, "-", "-"),
        "n_walks": r("Number of walks with readings in the segment.", "count", _TP, "-", "-"),
        "n_walks_start": r("Number of walks with start-window readings in the segment.", "count", _TP, "-", "-"),
        "n_readings": r("Number of readings in the segment (in p13_temperature_pairing_warmup: in that minute).", "count", _TP, "-", "-"),
        "mean_shade_matched": r("Segment mean of sensor-matched shade at arrival, at the association time constant, averaged over walks.", "fraction", _TP, "-", "-"),
        "mean_dose_1h_matched_wh_m2": r("Segment mean of sensor-matched 1 h sun dose, averaged over walks.", "Wh/m2", _TP, "-", "Clear-sky upper bound."),
        "mean_sky_view_factor_matched": r("Segment mean of sensor-matched sky view factor, averaged over walks.", "fraction", _TP, "-", "-"),
        "mean_height_width_ratio_matched": r("Segment mean of sensor-matched height-to-width ratio, averaged over walks.", "-", _TP, "-", "-"),
        # warm-up
        "minute": r("Whole minutes since the walk started.", "min", _TP, "floor(minutes_since_start)", "-"),
        "mean_start_departure_c": r("Mean of reading minus logger background minus the walk's mean after 15 minutes, per minute since start.", "°C", _TP, "-", "No allowance for position along the route."),
        "mean_start_departure_lo_c": r("Lower end of the interval of mean_start_departure_c.", "°C", _TP, _BOOT, "-"),
        "mean_start_departure_hi_c": r("Upper end of the interval of mean_start_departure_c.", "°C", _TP, _BOOT, "-"),
        "fitted_start_departure_c": r("Exponential settling curve fitted to mean_start_departure_c.", "°C", _TP, "amplitude × exp(-t / T), least squares weighted by readings", "-"),
        "adjusted_effect_c": r("Minute effect after walk offsets and position along the route (50 m bins per period) are allowed for.", "°C", _TP, "least squares two-way model, pooled periods", "Weakly identified: time and position move together."),
        "adjusted_effect_lo_c": r("Lower end of the interval of adjusted_effect_c.", "°C", _TP, _BOOT, "-"),
        "adjusted_effect_hi_c": r("Upper end of the interval of adjusted_effect_c.", "°C", _TP, _BOOT, "-"),
    }


def full_dictionary(radii=BUFFER_RADII_M, regimes: list[dict] | None = None) -> dict[str, dict]:
    """regimes: [{slug, name}] of the campaign-season wind regimes (rows for
    the regime-named point columns); without them those rows are absent."""
    regimes = regimes or []
    d = dict(_BASE)
    d.update(_V030)
    d.update(_V031)
    d.update(_regime_rows(regimes))
    d.update(_matched_rows(regimes))
    for k, v in p13_rows().items():
        d.setdefault(k, v)
    for template_id, template in _BUFFER_TEMPLATES.items():
        for r in radii:
            col_id = template_id.format(r=r)
            row = {k: (v.format(r=r) if isinstance(v, str) else v) for k, v in template.items()}
            row["status"] = "computed"
            d[col_id] = row
    for k, v in _SHADE_TABLE_ONLY.items():
        d.setdefault(k, v)
    for k, note in _RETIRED.items():
        d[k] = {**d[k], "status": f"RETIRED ({RETIRED_IN}): replaced by {note}"}
    return d


def dictionary_dataframe(radii=BUFFER_RADII_M, regimes: list[dict] | None = None):
    import pandas as pd

    d = full_dictionary(radii, regimes)
    rows = []
    for col_id, meta in d.items():
        row = {"id": col_id}
        row.update(meta)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    cols = ["id", "definition", "unit", "source", "method", "limits", "status"]
    return df[cols]
