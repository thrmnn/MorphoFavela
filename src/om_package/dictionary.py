"""P-08 — data dictionary: one row per variable ID, ever. IDs are never
reused; a variable retired in a later version keeps its row (marked
retired), it doesn't vanish. No variable has been retired yet.
"""
from __future__ import annotations

from .buffers import BUFFER_RADII_M
from .routes import ROUTE_FLAG_MAX_STREET_DIST_M
from .vent_indices import DEFAULT_BUFFER_M, MACDONALD_A, MACDONALD_BETA, MACDONALD_CD, VON_KARMAN

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
        "method": "nearest-neighbour join (<=20 m), scripts/brisa_ventilation/02_hw_canyon_proxy.py flanking-building cross-section (search radius = its SEARCH_RADIUS)",
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

# --- v0.2.0: P-10 (sun exposure) and P-11 (ventilation indices with
# time-matched wind). Every ventilation row is a PROXY from building
# geometry; the wind rows are SBGL airport observations, not wind at the
# route; no row is a measured air temperature or airflow. ---------------
_GEOM = "2019 building geometry (the geometry epoch is a build parameter)"
_SUN_PROXY = (
    "Geometry-derived (building and terrain horizon vs. sun position), not measured sunlight: "
    "no cloud, no tree shade."
)
_PREVAILING = (
    "evaluated at the prevailing wind bearing (frequency-weighted circular mean of the 2015-2024 SBGL "
    "climatology, data/maré/wind_rose.json; the same bearing P-06 uses)"
)
_VENT_LIMITS = (
    "PROXY, not simulated or measured air movement. SBGL (Galeão airport) is a regional reference, not wind at the "
    "route; a circular mean of a spread wind rose is a summary bearing, not a mode."
)
_BASE.update({
    "annual_sun_hours": {
        "definition": "Hours per year with the sun above both the geometric horizon and the marched building/terrain horizon at this point (static, like sky_view_factor). Geometry-derived PROXY for direct-sun exposure, not measured sunlight.",
        "unit": "h per year",
        "source": f"{_GEOM}; marched horizon (p10_horizon_profiles.parquet); pvlib solar position",
        "method": "10-min steps over one calendar year in Rio local time (America/Sao_Paulo); step counted sunlit when sun altitude > 0 and > the horizon angle at the sun's azimuth (nearest marched azimuth); hours = sunlit steps x step length (src/om_package/sun_envelope.py annual_sun_hours)",
        "limits": _SUN_PROXY + " Horizon march is limited to the DTM's valid radius (see README Known limits), so very distant obstructions are not seen.",
        "status": "computed",
    },
    "windward_lambda_f_prevailing": {
        "definition": "PROXY: frontal-area density facing the prevailing wind (Oke 1988 lambda_f of the nearest 10 m grid cell), " + _PREVAILING + ".",
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
        "definition": "PROXY: aerodynamic roughness length z0 by Macdonald et al. (1998), from the 50 m buffer's plan density and mean building height and the windward frontal-area density, " + _PREVAILING + ".",
        "unit": "m",
        "source": f"lambda_p_buffer_{DEFAULT_BUFFER_M}m, building_height_mean_buffer_{DEFAULT_BUFFER_M}m, windward_lambda_f_prevailing of this table",
        "method": f"Macdonald, Griffiths & Hall (1998), Atmos. Environ. 32(11):1857-1864: z0/H = (1 - zd/H) exp(-[0.5 beta (Cd/kappa^2) (1 - zd/H) lambda_f]^-0.5), A={MACDONALD_A:g}, beta={MACDONALD_BETA:g}, Cd={MACDONALD_CD:g}, kappa={VON_KARMAN:g} (src/om_package/vent_indices.py macdonald_zd_z0)",
        "limits": _VENT_LIMITS + " Staggered-array constants applied to an irregular favela fabric; NaN where the 50 m buffer has no building (mean height undefined).",
        "status": "computed",
    },
    "zd_macdonald_m": {
        "definition": "PROXY: displacement height zd by Macdonald et al. (1998), from the 50 m buffer's plan density and mean building height.",
        "unit": "m",
        "source": f"lambda_p_buffer_{DEFAULT_BUFFER_M}m and building_height_mean_buffer_{DEFAULT_BUFFER_M}m of this table",
        "method": f"zd/H = 1 + A^(-lambda_p) (lambda_p - 1), A={MACDONALD_A:g}; zd = (zd/H) x H (src/om_package/vent_indices.py macdonald_zd_z0)",
        "limits": _VENT_LIMITS + " Does not depend on wind direction. NaN where the 50 m buffer has no building.",
        "status": "computed",
    },
    "open_space_fraction": {
        "definition": "PROXY: share of the 50 m circular buffer not covered by building footprints (1 - lambda_p).",
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
        "method": "slot grid over the 24 h day; envelope table at 5 min, dose table on its own (coarser) grid stated in manifest.json p10.dose_slot_min",
        "limits": "Local time, not the device clock: whether the Octopus device clocks log UTC or local time is UNKNOWN (decision om_dates_tz). Map a device timestamp to a slot under both readings (clock_readings in src/om_package/sun_envelope.py; OM2/join_shade_example.py states the UTC-labelled P-05 convention).",
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
        "definition": "Which case a P-10 row describes: a campaign date (YYYY-MM-DD), envelope_min / envelope_median / envelope_max (statistic over every day of the window), or 'all' (clock-agreement row pooled over the campaign dates).",
        "unit": "category",
        "source": "src/om_package/p10_p11.py",
        "method": "campaign dates are read off the campaign CSVs (p05b_campaign_windows)",
        "limits": "A campaign date here is a calendar date in Rio local time; the device clock reading is unresolved (see local_slot).",
        "status": "computed",
    },
    "dose_1h_wh_m2": {
        "definition": "Clear-sky direct-beam dose on the horizontal plane over the 1 h up to and including this local slot (same day), zero while the point is shaded by the building horizon. Geometry-derived UPPER BOUND, not measured radiation.",
        "unit": "Wh/m2 (rounded to 0.1)",
        "source": f"pvlib Ineichen clear-sky DNI x sin(sun altitude); p10_horizon_profiles.parquet ({_GEOM})",
        "method": "per-slot beam energy = DNI x sin(altitude) x slot length, zero when shaded or sun down; trailing sum over 1 h (src/om_package/sun_envelope.py direct_sun_dose). Rows with scope = a date use that date; envelope_* rows are min/median/max over the window. Slots where the sun is down on every day and all doses are zero are omitted.",
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
        "method": "max over the Tregenza patches sharing the azimuth of the marched obstruction angle, march radius max_dist_m (README Known limits)",
        "limits": "Geometry only (no vegetation); limited to the march radius; cell resolution of the obstruction surface (1 m resampled from the DTM's native resolution).",
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
        "definition": "Observation time of an SBGL (Galeão airport) METAR report, UTC.",
        "unit": "ISO 8601, UTC", "source": "Iowa Environmental Mesonet ASOS archive, station SBGL (provenance.wind_source in manifest.json)",
        "method": "as reported", "limits": "SBGL is a regional reference at 10 m, not wind at the route.",
        "status": "computed",
    },
    "valid_local": {
        "definition": "valid_utc converted to Rio local time (fixed UTC-3).",
        "unit": "ISO 8601, local time (no offset)", "source": "valid_utc", "method": "tz conversion to America/Sao_Paulo",
        "limits": "Offset from the tz database, not typed.", "status": "computed",
    },
    "drct": {
        "definition": "Wind direction the wind blows FROM at SBGL, degrees clockwise from north; empty for calm or variable reports.",
        "unit": "degrees", "source": "SBGL METAR (Iowa ASOS archive)", "method": "as reported",
        "limits": "Observed at the airport, not at the route. Reported on a coarse direction grid by the source.",
        "status": "computed",
    },
    "speed_ms": {
        "definition": "Wind speed at SBGL, 10 m.",
        "unit": "m/s", "source": "SBGL METAR (Iowa ASOS archive), reported in knots",
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
        "method": "for each 5-min step of a campaign walk window, the nearest SBGL report with a usable direction within the match gap (manifest.json p11.max_gap_min) (src/om_package/wind_obs.py wind_at)",
        "limits": "The device clock reading is UNKNOWN; this column and used_if_device_clock_local are the two readings. Time-matched wind is SBGL, not at the route.",
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

_DESCOPED_STATUS = "DESCOPED (om_v013_descope)"

_DESCOPED: dict[str, dict] = {
    "sky_view_factor_terrestrial": {
        "definition": "Terrestrial (ground-instrument) sky-view factor at each OM2 point.",
        "unit": "fraction [0,1]", "source": "DESCOPED — terrestrial-LiDAR analysis is out of scope in this version (descoped from v0.1.3; PI decision om_v013_descope)",
        "method": "DESCOPED", "limits": "No column in this version (spec P-04: airborne only). May come in a later version.", "status": _DESCOPED_STATUS,
    },
    "tree_shade": {
        "definition": "Whether tree canopy shades each OM2 point. RESERVED column in the shade table schema (SHADE_TABLE_COLUMNS) — present but always null, so the table's shape will not change if it is added later.",
        "unit": "bool", "source": "DESCOPED — tree shade is out of scope in this version (descoped from v0.1.3; PI decision om_v013_descope)",
        "method": "DESCOPED", "limits": "Always null in this version (reserved column). Building-only shade is the 'shaded' column. May come in a later version.", "status": _DESCOPED_STATUS,
    },
    "airborne_vs_terrestrial_comparison": {
        "definition": "Comparison of airborne vs. terrestrial form-variable estimates along OM2.",
        "unit": "-", "source": "DESCOPED — terrestrial-LiDAR analysis is out of scope in this version (descoped from v0.1.3; PI decision om_v013_descope)",
        "method": "DESCOPED", "limits": "Not computed in this version. May come in a later version.", "status": _DESCOPED_STATUS,
    },
    "height_change_2024_2026": {
        "definition": "Change in building/canopy height between the 2024 airborne LiDAR and the 2026 OM2 terrestrial field campaign.",
        "unit": "m", "source": "DESCOPED — terrestrial-LiDAR analysis is out of scope in this version (descoped from v0.1.3; PI decision om_v013_descope)",
        "method": "DESCOPED", "limits": "Not computed in this version; the name may be revisited if it comes in a later version.", "status": _DESCOPED_STATUS,
    },
}

_SHADE_TABLE_ONLY = {
    "timestamp": {"definition": "Clock timestamp of a shade evaluation (5-min step).", "unit": "datetime, LABELLED UTC in v0.1.2 (a stated operating-rule choice, NOT a resolution of the still-UNRESOLVED campaign timezone — see src/om_package/shade.py module docstring; tz is a required, no-default parameter of every shade/join function)", "source": "src/om_package/shade.py", "method": "pd.date_range over the requested time window", "limits": "-", "status": "computed"},
    "date": {"definition": "Calendar date of a shade evaluation.", "unit": "date", "source": "src/om_package/shade.py", "method": "-", "limits": "-", "status": "computed"},
    "sun_altitude_deg": {"definition": "Apparent solar elevation at the evaluation timestamp.", "unit": "degrees", "source": "pvlib.solarposition.get_solarposition", "method": "-", "limits": "-", "status": "computed"},
    "sun_azimuth_deg": {"definition": "Solar azimuth (clockwise from north) at the evaluation timestamp.", "unit": "degrees", "source": "pvlib.solarposition.get_solarposition", "method": "-", "limits": "-", "status": "computed"},
    "shaded": {"definition": "True when the point gets no direct sun at this timestamp: a building (not tree) blocks the sun, or the sun is below the horizon (night, sun_altitude_deg <= 0). Night rows are no direct sun, not building shade: take building-shade shares over rows with sun_altitude_deg > 0 only.", "unit": "bool", "source": "src/om_package/shade.py is_shaded()", "method": "sun altitude vs. marched horizon angle at the sun's azimuth (point_horizon_profiles(), wired v0.1.2, max_dist_m=100m — see README Known limits)", "limits": "v0.1.2 covers only the pilot's 5 campaign dates (one CSV per device pulled 2026-09-25); more dates arrive as more CSVs are pulled. tz='UTC' labelling, not a resolved local time.", "status": "computed"},
}


def full_dictionary(radii=BUFFER_RADII_M) -> dict[str, dict]:
    d = dict(_BASE)
    for template_id, template in _BUFFER_TEMPLATES.items():
        for r in radii:
            col_id = template_id.format(r=r)
            row = {k: (v.format(r=r) if isinstance(v, str) else v) for k, v in template.items()}
            row["status"] = "computed"
            d[col_id] = row
    for k, v in _DESCOPED.items():
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
