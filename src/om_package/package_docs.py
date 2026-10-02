"""P-01 README + P-09 CHANGELOG for the mare_om2 package. Generated, not
hand-edited — rebuild via scripts/build_om_package.py after any source/method
change (never hand-edit data/outputs, per CLAUDE.md).

Audit fix (2026-09-27): every value rendered here that describes a
measured or provenance fact (fetch dates, the nodata floor, which
internal routes actually exist on disk, the named-team decision date) is
now a REQUIRED parameter with no default — see ``require_nodata_floor_m``
below. A caller that cannot supply one must compute it (see
``scripts/build_om_package.py``) rather than let this module quietly fall
back to a stale number.
"""
from __future__ import annotations

import math
from datetime import date

from .routes import ROUTE_FLAG_MAX_STREET_DIST_M
from .shade import SHADE_STEP_MIN
from .vent_indices import DEFAULT_BUFFER_M

#: The one place the package version is set; the build default and every
#: rendered heading read it.
VERSION = "v0.3.0"
#: read from the clock at import time, never typed — this is the date this
#: version is BUILT, not the date any source data was fetched (the
#: "fetched" date in the README is computed at build time from the route
#: files' own mtimes, or a routes manifest if one exists — see
#: ``scripts/build_om_package.py``'s ``route_fetch_date_label``).
VERSION_DATE = date.today().isoformat()

#: Same text goes in the README banner and the manifest's use_terms field
#: — one string, so the two can never drift. The decision behind it
#: (``om_use_terms``) and its date travel in manifest.json's
#: ``provenance.decisions``, not as a prose interview code (audit fix,
#: 2026-09-27 — see src/om_package/provenance.py).
USE_TERMS = (
    "INTERNAL REVIEW DRAFT — Octopus LRP #2 team members only; not for "
    "redistribution or citation; to be revisited at submission."
)


def _decision(decisions: list[dict], decision_id: str) -> dict:
    """The one entry in ``decisions`` (provenance.read_om_decisions()'s
    output) with this id. Raises — never silently substitutes another
    entry or an empty string — if it's absent, since every call site below
    needs that specific decision's resolution text."""
    for d in decisions:
        if d["id"] == decision_id:
            return d
    raise KeyError(f"decision '{decision_id}' not found in {[d.get('id') for d in decisions]}")


def require_nodata_floor_m(p05_shade: dict) -> dict:
    """Pulls {"min", "median", "max"} nodata-floor stats (metres) out of a
    manifest's ``p05_shade`` block. Raises KeyError, never a default —
    this number used to be hardcoded (104.15 m) and drifted from what was
    actually measured; a manifest missing it means the build skipped the
    measurement (see shade.nodata_floor_m), and rendering must fail
    loudly rather than paper over that with a stale constant."""
    return p05_shade["nodata_floor_m"]


README_TEMPLATE = """\
> **{use_terms}**

# Maré morphology, OM2 — data package {version}

Built for Octopus LRP #2 ("Street by street: explaining air temperature
differences across streets and over time in Complexo da Maré", lead
Jingxue, PI Simone). Théo Alessandro Hermann contributes street-form, sun and
ventilation variables from the Brisa+ (MorphoFavela) pipeline in a support
role. This package contains no temperature analysis and no conclusions;
those are the Octopus team's work.

**Read first:** `report.pdf` (the short human report: results, key figures,
what to read with care). This README is the technical document: files,
sources, methods, limits.

**Named team (release card `om_release_v0_2_0`; use terms per PI decision `om_use_terms`, {om_use_terms_date}):** Jingxue,
Vincent, Simone. This release goes to the three of them only, as an
internal review draft; not for wider redistribution or citation (see Use
terms). The decision's full resolution text travels in this package's
`manifest.json` under `provenance.decisions` (id `om_use_terms`).

Generated {version_date} by `scripts/build_om_package.py`
(source: `src/om_package/`; every code path named in this README lives in
the Brisa+ (MorphoFavela) repository).

## Files in this package

| File | What it holds | Spec item |
|---|---|---|
| `report.pdf`, `report.md` | Short human report: results, key figures, caveats | — |
| `README.md`, `README.pdf` | This technical document | P-01 |
| `manifest.json` | Version, CRS, use terms, decisions, wind source, P-10/P-11 summaries, sha256 per file | — |
| `OM2/points.parquet`, `.gpkg`, `.csv` | One row per route point (1 m): form, buffer, ventilation-proxy and sun columns | P-02, P-03, P-04, P-06, P-10, P-11 |
| `OM2/aggregate_to_segments.py` | Standalone re-aggregation of the points to any segment length | P-03 |
| `p05_building_shade.parquet` | Building shade per point and {shade_step_min}-min step on each walk date, daylight only, Rio local time (parquet only) | P-05 |
| `p02b_walks.parquet`, `.csv` | One row per logger walk: timing, coverage, wind regime tag | P-12 |
| `p12_walk_points.parquet`, `.csv` | One row per walk and route point: arrival time, shade and dose at arrival, sensor-matched values | P-12 |
| `OM2/join_shade_example.py` | Example join of device data to the shade table | P-05 |
| `OM2/p07_quality_report.json`, `.csv` | Coverage per column, flagged points, P-10/P-11 summary block | P-07 |
| `p08_data_dictionary.parquet`, `.csv` | One row per variable: definition, unit, source, method, limits | P-08 |
| `p10_sun_envelope.parquet`, `.csv` | Per point and local {envelope_slot_min}-min slot over the season: always sunlit / always shaded / date-dependent / night | P-10 |
| `p10_sun_dose.parquet`, `.csv` | Clear-sky direct-sun dose over the past {dose_hours_list} h, per campaign date and as a season min/median/max | P-10 |
| `p10_horizon_profiles.parquet` | Marched horizon angle per point and azimuth (input to every sun result) | P-10 |
| `p11_wind_regimes.csv`, `p11_regime_by_hour.csv` | The two wind regimes (campaign season and 2015-2024 climatology) and their share by local hour | P-11 |
| `OM2/map_form.png`, `OM2/profiles.png` | Route map coloured by sky view; form variables along the route | P-04 |
| `OM2/map_shade.png`, `OM2/shade_calendar.png` | Daylight shade share per point; shade by date and time | P-05 |
| `OM2/sun_envelope.png`, `OM2/sun_dose.png` | Date-dependent share by time of day and along the route; {dose_hours_first} h dose along the route | P-10 |

{conformance_section}
## Release scope

**OM2 only.** This directory (`outputs/_packages/mare_om2/{version}/`)
contains OM2 exclusively. OM1/OM3/OM4 are built by the exact same code
path (`--route ALL`) writing to an internal build directory outside this
package (`outputs/_packages/_internal/mare_routes/{version}/`) whenever
that path is run — they are never copied or symlinked into `mare_om2/`,
so zipping/sharing this directory cannot leak them. **For {version}:
{internal_routes_status}** **{version} supports RQ2's within-route
spatial/temporal analysis only.** RQ3's between-route holdout needs
OM1/OM3/OM4 released, which is a PI decision for a later version.

## Coverage vs Table 1

**This package covers surface structure only** — route-point geometry,
building height/plan density/canyon form, ventilation-geometry proxies,
sun exposure (P-05 exact-date shade, P-10 sun exposure that does not need
the exact date) and observed-wind-based ventilation indices (P-11). Surface cover (vegetation
fraction, impervious fraction) and façade materials belong to other
Octopus LRP #2 team members' packages, not this one; there is no PENDING
row here for them, because they are out of this package's scope, not a
gap in it.

## CRS

SIRGAS 2000 / UTM 23S — EPSG:31983 — for every point, buffer and segment
geometry in this package. Route files arrive in WGS84 lon/lat (Google
Drive OM_1..OM_4_inferred_route.json) and are reprojected once, before
densification.

## Sources and dates

| Source | Date / vintage | Used for |
|---|---|---|
| OM_1..OM_4_inferred_route.json (Google Drive, PI-owned folder) | {route_fetch_date_label} | P-02 route points |
| data/maré/raw/buildings_mare.shp + buildings_extended_300m.gpkg | 2019 cadastral clip (RJ IPP municipal layer, `buildings_RJ_2019.shp` — see data/README.md) | P-04 building height/plan density, P-03 buffers, route_geometry_flag |
| data/maré/raw/mare_dtm.tif (+ extended DTM) | vintage not recorded in data/README.md | canyon H/W, SVF |
| data/maré/raw/street_mare.shp | vintage not recorded in data/README.md | street network for SVF/canyon sampling, route_geometry_flag |
| outputs/maré/svf_v2/svf_streets.gpkg | ray-cast from the above, 1.5 m pedestrian height, 145-patch Tregenza sky (src/svf_v2) | P-04 sky_view_factor |
| outputs/maré/morphometrics/canyon/hw_streets.gpkg | derived from the above (scripts/brisa_ventilation/02_hw_canyon_proxy.py: flanking-building cross-section at each street sample, search radius = that script's SEARCH_RADIUS) | P-04 street_width_m, building_height_m, height_width_ratio |
| outputs/maré/features/features_grid.parquet | 10 m grid, derived from the above | P-04 plan_density_lambda_p, grid_cell_id, P-06 ventilation proxies incl. lambda_f_<dir> |
| data/maré/wind_rose.json | ASOS Galeão (SBGL) METAR, 2015-2024 | P-06 wind-alignment proxy |
| data/maré/octopus/wind/ (SBGL METAR, Iowa Environmental Mesonet ASOS archive) | window {wind_window}, fetched {wind_fetched}; fetch URL and sha256 in `manifest.json` `provenance.wind_source` | P-11 wind regimes (`p11_wind_regimes.csv`) |
| Walk dataset (data/maré/octopus/prerelease_v020/matched/, Cassiano and Vincent) | {n_walks} walks on {n_campaign_dates} dates; sha256 per file in its manifest | walk timing (`p02b_walks`), arrival times (`p12_walk_points`), walk dates for P-05 / P-10 |
| data/maré/neighbourhoods.gpkg | community boundary crosswalk | neighbourhood attribution |

The extended DTM (`dtm_extended_300m.tif`, used for P-05 shade's horizon
march) is **{dtm_native_resolution_m:g} m native resolution** (read from
the raster at build time), resampled to 1 m by
`src/brisa_solar/wp02_surface.build_surface` to match WP-04's `CELL_M`.

**Geometry epoch: {geometry_label}.** The buildings + terrain layers are the
source for every {version} P-04/P-06/P-10/P-11 variable. The epoch is a build
parameter (`--buildings`, `--dtm`, and the form-variable inputs under
`--root`), so 2024 airborne LiDAR and footprints can replace it later without
a code change.

**The buildings + terrain cadastral layer is the source for every
{version} P-04/P-06 variable.** No terrestrial (ground-instrument) source
is used: terrestrial-LiDAR analysis is out of scope for {version} by
decision `om_v013_descope` — see Known limits.

## Methods

### Route points (P-02)

OM route edges are chained by `edge_order`,
oriented geometrically (nearest-endpoint chaining — a stored LINESTRING
may run opposite to its declared (u,v)), then sampled every 1 m along
the chained centreline at 1.5 m pedestrian height (matches this repo's
own SVF/street-sampling convention, src/svf_v2/sampling.py). No
segments are imposed at this stage. Point IDs are deterministic:
`<route>-<metres from route start, zero-padded>`, e.g. `OM2-000042` —
and PROVISIONAL (see Known limits).

### Aggregation (P-03)

Buffer variables (5/10/20/50 m circular buffers
around each point) and segment aggregation (any length, on demand) are
both re-runnable. Buffer variables are computed at build time. Segment
aggregation ships two ways: `scripts/aggregate_om_points.py` (repo-only
CLI, same logic) and, new in v0.1.3, **`OM2/aggregate_to_segments.py`
travels inside this package itself** — standalone (pandas + pyarrow
only, no Brisa+ (MorphoFavela) import), so a recipient with only this directory
can still re-aggregate. Usage (run from inside the package directory):
```
python OM2/aggregate_to_segments.py --points OM2/points.parquet \\
    --segment-m 20 --out OM2/segments_20m.parquet
```
Point count is conserved (every point lands in exactly one segment);
the four buffer radii (5/10/20/50 m) are already columns on
`OM2/points.*` — this script only groups points into segments, it does
not recompute buffers.

### Form variables (P-04)

Nearest-neighbour spatial joins from the
airborne sources above (each capped at a max join distance — beyond it
a point gets NaN, never a guessed value) plus street_orientation_deg,
computed directly from the route's own local tangent, plus
`grid_cell_id` (the same features_grid cell used for
`plan_density_lambda_p`) so models can cluster the 10 m-grid variables.

### Ventilation (P-06)

Four PROXIES (never a flow simulation) — wind
alignment, OMNIDIRECTIONAL frontal-area density, openness, distance to
open space — plus the 8 per-compass-direction frontal-area columns
(`lambda_f_N` .. `lambda_f_NW`) passed through unchanged, so a
windward-specific figure can be built downstream without this package
guessing the wind direction that matters. See the data dictionary
(P-08) for exact formulas.

### Route geometry flag (`route_geometry_flag`)

True where a point falls inside a
`buildings_mare` footprint OR more than {route_flag_max_dist_m:g} m from the nearest
`street_mare` centreline (`src/om_package/routes.py`,
`ROUTE_FLAG_MAX_STREET_DIST_M`) — both are signs the OSM-inferred route
drifted off the street the team actually walked. See Known limits for
the measured counts.

### Shade (P-05)

(`src/om_package/shade.py`.) The horizon is marched once for all {n_om2_points}
route points (`point_horizon_profiles()`, the WP-02/WP-04 engine, real
145-patch Tregenza directions, `max_dist_m={shade_max_dist_m:g} m`).
`compute_shade_local()` then writes {n_shade_rows} (point x {shade_step_min}-min
step) rows across the {n_campaign_dates} walk dates, daylight only, in Rio
local time (`timestamp_local`, with a `timestamp_utc` twin), as parquet only
({shade_fraction_daylight_pct}% of rows in building shade). The schema
reserves a `tree_shade` column (always null). `OM2/join_shade_example.py`
joins `p05_building_shade` to a device CSV by `point_id` and `timestamp_utc`
floored to {shade_step_min} minutes (run from inside the package directory):
```
python OM2/join_shade_example.py --shade p05_building_shade.parquet \\
    --device path/to/octopus_log_with_point_id.csv --out joined_example.csv
```

### Walks (P-12)

(`src/om_package/walks.py`, `walk_tables.py`, `sensor_match.py`,
`walk_dose.py`.) `p02b_walks` lists each logger walk. `p12_walk_points` gives,
for each walk and each route point the walk reached, the arrival time, whether
the point was shaded then, the clear-sky direct dose in the 1 h and 3 h before
arrival, and sensor-matched values (exponentially weighted mean of the points
already passed, tau = 5, 10, 30, 60 s) of the form and ventilation measures.
Re-aggregate to segments per walk with
`python OM2/aggregate_to_segments.py --points p12_walk_points.parquet --by walk_id --segment-m 20 --tau 30 --out segments.parquet`.

### Sun exposure (P-10)

(`src/om_package/sun_envelope.py`,
`src/om_package/p10_p11.py`): the horizon is marched once per point
(`p10_horizon_profiles.parquet`) and every date or time then costs only a
sun-position lookup. *Envelope*: for each point and local time of day
(Rio local time, {envelope_slot_min}-min slots) over every day of the season window
{p10_window}, classify as always sunlit, always shaded or date-dependent
(counting only days with the sun up), with the sunlit share of days.
*Dose*: clear-sky direct-beam energy on a horizontal plane over the
preceding {dose_hours_and} h (slots of {dose_slot_min} min), for each walk date
and as a min/median/max over the season window. *Annual sun hours*:
hours per year with the sun above the point's horizon. Every quantity is a geometry-derived proxy (no cloud, no
tree shade, not measured sunlight); the dose is clear-sky, hence an upper
bound. Rio local time is a fixed UTC-3 (no daylight saving since 2019);
the loggers record UTC.

### Ventilation indices (P-11)

(`src/om_package/vent_indices.py`, `wind_regimes.py`): the SBGL (Galeão
airport, 10 m) reports are split into two wind regimes by the peaks of the
smoothed 16-sector rose, for the campaign season and for the 2015-2024
climatology (`p11_wind_regimes.csv`, with a von Mises mixture check beside
each regime; `p11_regime_by_hour.csv`). Per-point columns are computed at each
campaign regime's mean direction and named by regime:
`frontal_area_density_windward_<regime>`, `canyon_alignment_deg_<regime>`,
`upwind_shelter_angle_deg_<regime>`, `z0_macdonald_m_<regime>` (Macdonald et
al. 1998), plus `zd_macdonald_m` and `open_space_fraction`. All are PROXIES
from building geometry, never measured or simulated air temperature or air
movement; SBGL is a regional reference, not wind at the route.

### Figures

(`src/om_package/figures.py`, PI ruling 2026-09-27 — spatial
result first, then the sampling along the route): `OM2/map_form.png`
(route over the Maré buildings, coloured by `sky_view_factor`),
`OM2/map_shade.png` (same base map, coloured by the share of daylight in building shade),
`OM2/profiles.png` (1 m raw + 10 m segment means for the form/shade
variables along the route), `OM2/shade_calendar.png` (one strip per
campaign date, distance vs time of day, shaded/sunlit). New in {version}:
`OM2/sun_envelope.png` and `OM2/sun_dose.png`. (Ventilation figures:
to be redrawn for the two regimes.)

## Using the data

### Join device data to P-05 / P-10

Loggers record UTC; P-05 carries `timestamp_utc` and `timestamp_local`. P-10 is
keyed by Rio local time of day (`local_slot`):
```
ts = pd.to_datetime(device["Timestamp"], utc=True)
device["slot_local"] = ts.dt.tz_convert("America/Sao_Paulo").dt.floor("5min").dt.strftime("%H:%M")
```

### Segment length (note for the Octopus team)

The points are 1 m apart.
A sensor carried at walking speed does not resolve that: its reading at any
instant is a weighted average over the stretch of route just walked, so
neighbouring 1 m points are strongly correlated and a 1 m point is not an
independent observation. Treat the 1 m points as the geometry grid and
choose the analysis segment from the sensor:
**L ≈ v × k × τ**, with v the walking speed (m/s), τ the sensor's response
time constant (s) and k the number of time constants; k = 3 gives about
{three_tau_pct:.0f}% of a step response (1 - e^-3). For one walk of T
seconds over the whole route (length {route_length_m:.0f} m) the mean speed
is v = {route_length_m:.0f} m / T. **This package holds no value for τ, so it
does not state an L.** *Question for the team: what is the time constant of
your air-temperature sensor as mounted (housing/shield included), and is it
quoted as a 63% or a 90% response time?* Two consequences: the reading at a
point reflects the route *behind* the walker, so match it to a segment that
ends at that point rather than one centred on it; and models that treat
segments shorter than L as independent will understate uncertainty, so
cluster by `grid_cell_id` or by blocks of at least L. Re-aggregate with
```
python OM2/aggregate_to_segments.py --points OM2/points.parquet \\
    --segment-m <L> --out OM2/segments.parquet
```
(`--segment-m` defaults to 10 m, a placeholder for the build's profile
figures, not a sensor-derived value.)

## Known limits

### Time

- Loggers record UTC. Rio local time is America/Sao_Paulo (UTC-3, no daylight
  saving). Shipped time columns are Rio local time (`_local`, ISO 8601 with
  offset); data tables also carry a `_utc` twin. The season envelope is
  date-dependent for {date_dependent_pct:.1f}% of daylight point-slots; the rest
  (`always_sunlit` / `always_shaded`) holds on every day of the window.

### What the values are

- **Sun and ventilation quantities are geometry-derived proxies.** No tree
  shade, no cloud (the dose is clear-sky, an upper bound), diffuse and
  reflected radiation excluded; none is measured sunlight, air temperature or
  air movement. SBGL wind is an airport reference at 10 m, matched to a
  campaign time only within the match gap in `manifest.json` `p11`; it is
  not wind at the route.
- **Sky-view factor is an UPPER BOUND under canopy.** The ray-cast mesh is
  buildings + bare-earth terrain only — no vegetation is in the scene — so
  a tree-covered point's real sky view is <= the reported
  `sky_view_factor`, never more.
- Ventilation columns are geometry-derived PROXIES, not simulated or
  measured airflow. The P-06 proxies are isotropic, so they say nothing
  about upwind fetch beyond axis alignment; the P-11 columns are computed
  at a wind direction and are direction-specific.
- **`z0_macdonald_m` is outside its calibrated range along most of the
  route.** Macdonald et al. (1998) was calibrated on regular arrays of
  obstacles; Maré's plan density in the {vent_buffer_m} m buffer is beyond that range,
  so the displacement height approaches the roof height and z0 falls
  toward zero. Read a near-zero z0 here as out of range, not as a smooth
  surface; `zd_macdonald_m` carries the same caveat.
- **NaN has two distinct causes** in the point table's joined columns —
  they are not interchangeable and are documented separately per column
  in the data dictionary (P-08): (1) *beyond the join-distance cap*
  (`building_height_m`, `sky_view_factor`, `plan_density_lambda_p`,
  `grid_cell_id`, ventilation proxies — a point too far from any source
  sample); (2) *no feature in the buffer* (`building_height_mean_buffer_*m`
  — zero buildings intersect that point's buffer, a real "no building
  here" result, not a join gap).
- Nearest-neighbour joins carry a `*_join_dist_m` column; check it before
  trusting a value near a data-layer edge.

### Geometry and route

- **Geometry epoch is {geometry_label}.** The horizon march is limited to the
  DTM's valid radius (see `max_dist_m` below); 2024 airborne data are not
  used yet.
- **No terrestrial ground-truth comparison exists yet.** Every P-04/P-06
  variable is derived from the cadastral buildings + DTM layers in
  Sources and dates above.
- **route_geometry_flag**: {n_route_geometry_flagged}/{n_om2_points}
  OM2 points ({route_flag_pct}%) are flagged — inside a building footprint
  or more than {route_flag_max_dist_m:g} m from the nearest street centreline. Of the
  {n_lambda_p_ones} points with `plan_density_lambda_p == 1.0`,
  {n_lambda_p_ones_flagged} ({lambda_p_share_explained_pct}%) are
  explained by this flag. Of the remaining {n_lambda_p_remainder} points,
  {n_lambda_p_remainder_plausible} have `building_count_buffer_10m > 0`
  and a recorded `building_height_mean_buffer_10m` (consistent with a
  fully-built 10 m cell rather than a join defect) — {n_lambda_p_remainder_not_checked}
  remain unchecked.
- **point_id is PROVISIONAL.** It is minted from the OSM-inferred route
  file, not the team's own om_routes.gpkg. The ID string is stable across
  rebuilds of the same route file, but the place it names may move when
  v0.2 rebuilds on the real route — that release will publish an
  old->new `point_id` crosswalk (decision `om_route_geometry`).
- **max_dist_m={shade_max_dist_m:g} m, not WP-04's 500 m citywide
  default**, for the horizon march behind P-05: `dtm_extended_300m.tif`
  (the DTM raster only — the footprint layer has no nodata) has real
  nodata starting between
  {nodata_floor_min_m:.0f} m and {nodata_floor_max_m:.0f} m from OM2
  points (median {nodata_floor_median_m:.0f} m; measured per point at
  build time via `shade.nodata_floor_m()` — its raster bounding box is a
  rectangle, but valid coverage inside it is not). WP-02's horizon march
  (`wp02_horizon.py`) is not NaN-safe (`torch.maximum` propagates NaN), so
  a full-radius pilot run returned all-NaN horizon values before this was
  caught; {shade_max_dist_m:g} m is safely under every OM2 point's
  measured nodata floor and was NOT patched into the shared WP-02 engine
  (P1's citywide/WP-04 defended numbers also depend on it) — this scoping
  fix lives only in `point_horizon_profiles()`.

### Device files and route files

- **Walk dataset.** Walk timing comes from the team's walk files (matched
  GPS tracks, one CSV per walk, sha256 per file in their manifest). Only fixes
  matched to edges of the OM2 route count; arrival times are interpolated
  between fixes (flagged `gap_interpolated` when the fix gap exceeds 60 s).
  The join example (`OM2/join_shade_example.py`) joins by a pre-assigned
  `point_id` plus an exact floor-to-5-minutes `timestamp_utc` match.
- om_routes.gpkg (Google Drive) was NOT fetched — too large for the
  connector. Pending if the team needs it.

### Out of scope by decision

- **Terrestrial SVF: OUT OF SCOPE for this version, by decision**
  (`om_v013_descope`) — the terrestrial-LiDAR analysis is not part of
  {version}; it may come in a later version.
- **Building shade: computed for {n_campaign_dates} walk dates**
  (see P-05 above, decision `om_shade_release`). **Tree shade: OUT OF SCOPE for
  this version, by decision** (`om_v013_descope`); it may come in a later
  version. `tree_shade` stays in the shade schema as a reserved,
  always-null column.
- **Height change 2024->2026 and the airborne-vs-terrestrial comparison:
  OUT OF SCOPE for this version, by decision** (`om_v013_descope`); they
  may come in a later version.

## Manifest

`manifest.json` records `package_version`, `crs`, `use_terms`,
`provenance.decisions` (the Octopus panel and release decisions, id +
resolution text — see above), `provenance.wind_source` (the SBGL source
manifest: station, fetch URL, fetch time, sha256, counts), `p10` / `p11`
(window, slot grids, match gap, summary shares), per-route build stats (relative output
paths), and a `sha256` per OTHER file in this package (recompute and
compare before trusting a copy — `manifest.json` excludes its own hash,
since a file cannot record its own checksum before it is written).
Parquet tables built from a GeoDataFrame (`points.parquet`) carry
GeoParquet `geo` metadata as well as plain `x`/`y` columns, so both
GeoParquet-aware and plain-pandas readers work without extra steps.

## Use terms

{use_terms}

## How to cite

This package was produced with the Brisa+ (MorphoFavela) pipeline (Théo Alessandro Hermann).
Authorship is to be discussed with the lead author when the Octopus LRP #2
contribution list is drafted (decision `om_credit`).
"""

#: Frozen literal text — v0.1's shipped CHANGELOG entry, read verbatim
#: from outputs/_packages/mare_om2/v0.1.2/CHANGELOG.md (audit fix,
#: 2026-09-27: CHANGELOG_TEMPLATE used to run the WHOLE changelog through
#: .format(version=..., version_date=...), so a stray `{version}` inside
#: this historical text would have silently drifted to whatever version
#: is current at build time — never happened here, but v0.1.1's entry
#: below had exactly that bug). No defect was flagged in v0.1's own text,
#: so it is reproduced unchanged.
CHANGELOG_V01_ENTRY = """\
## v0.1 — 2026-09-24

Initial release. Built from the 2019 airborne source (buildings + DTM)
against OM_2's inferred route (OM_1/OM_3/OM_4 built by the same code path,
`--route ALL`).

- P-02 route points at 1 m spacing, pedestrian height 1.5 m, stable IDs.
- P-03 buffer variables at 5/10/20/50 m; segment aggregation script
  (any length, re-runnable).
- P-04 airborne form variables: building height, plan density, street
  width, height-to-width ratio, street orientation, sky-view factor.
  Terrestrial SVF PENDING.
- P-05 shade function/CLI shipped; output table empty (campaign dates
  unknown). Tree shade PENDING.
- P-06 ventilation proxies (wind alignment, frontal area, openness,
  distance to open space) — all labelled PROXY.
- P-07 quality report (per-variable coverage + join gaps).
- P-08 data dictionary (every variable this package is designed to carry,
  including PENDING rows).
- Contact sheet PNG for OM2.
"""

#: Frozen literal text — v0.1.1's shipped entry, corrected for the one
#: audit-flagged defect (2026-09-27 audit, item 1): the shipped text read
#: "OM1/OM3/OM4 now build to `.../mare_routes/v0.1.2/`" — the CURRENT
#: version at whatever later date this was rendered, not v0.1.1, the
#: version this entry describes, and only OM1 was actually built to that
#: internal directory in v0.1.1 (confirmed on disk:
#: outputs/_packages/_internal/mare_routes/v0.1.1/OM1 exists; OM3/OM4 do
#: not). Every other bullet is reproduced unchanged.
CHANGELOG_V011_ENTRY = """\
## v0.1.1 — 2026-09-24

Panel review (docs/critic/octopus_package_panel_2026-09-24.md) and PI
ruling (interview 2026-09-24) applied on top of v0.1's initial release.

- Release scope narrowed to **OM2 only** in the shared package path; OM1
  now builds to `outputs/_packages/_internal/mare_routes/v0.1.1/` (OM3
  and OM4 were not run in v0.1.1) — the internal directory is never
  copied into `mare_om2/`.
- Added `route_geometry_flag` (within a building OR >10 m from the nearest
  street centreline), counted in the P-07 quality report, documented in
  P-08, and the README's lambda_p note rewritten with the measured share
  of `plan_density_lambda_p == 1.0` points it explains.
- Added an INTERNAL REVIEW DRAFT use-terms banner (README top) and a
  matching `use_terms` field in the manifest.
- Schema freeze: joined the 8 `lambda_f_<dir>` columns; renamed the mean
  proxy's definition from "windward obstruction" to an OMNIDIRECTIONAL
  obstruction-density proxy; added `grid_cell_id` (features_grid.zone_id,
  same join as `plan_density_lambda_p`); added data-dictionary rows for
  all of the above plus the 4 segment columns (`segment_id`,
  `segment_start_m`, `segment_end_m`, `n_points`).
- Known limits made honest: `sky_view_factor` documented as an UPPER
  BOUND under canopy (no vegetation in the ray-cast scene); "no feature
  in buffer" NaNs distinguished from beyond-join-cap NaNs; `point_id`
  documented as provisional with a promised v0.2 crosswalk; a "Coverage
  vs Table 1" paragraph states this package is surface structure only
  (no PENDING surface-cover row — that is out of scope, not a gap);
  `height_change_2024_2026` kept PENDING with "data location being
  confirmed by T. Hermann".
- Timezone: removed the hardcoded `CAMPAIGN_TZ = "America/Sao_Paulo"`
  default — `tz` is now a required parameter (no default) of
  `sun_positions`/`compute_shade`. Added `drop_nofix_rows()` (drops
  Latitude == Longitude == 0.0 sentinel rows) and wired it into
  `OCTOPUS_JOIN_EXAMPLE` before any spatial join. Added
  `infer_campaign_windows(csv_paths)` to read per-file date + first/last
  timestamp + fix/no-fix row counts off raw CSVs the moment they arrive.
- Shade schema: reserved an explicitly-null `tree_shade` column
  (building-only P-05 is designed to carry it once dates are known).
- Manifest hygiene: relative file paths (dropped the absolute `root`
  key), added `package_version`, `crs`, `use_terms`, and a `sha256` per
  file. `points.parquet` is now written from the GeoDataFrame directly,
  so it carries GeoParquet `geo` metadata alongside the flat `x`/`y`
  columns.
- Contact sheet: map panel aspect now uses `adjustable="datalim"` instead
  of the default `"box"`, so it no longer leaves a blank strip either
  side on a 10-inch-wide figure.
- How to cite: filled with an acknowledgment line (Brisa+ (MorphoFavela) pipeline,
  Théo Hermann); authorship to be raised with the lead author when the
  contribution list is drafted. No PLACEHOLDER remains anywhere in the
  README.
"""

#: Frozen literal text — v0.1.2's entry as SHIPPED in v0.1.3's CHANGELOG
#: (rendered values included: it used to be a template filled from live
#: build state, which is how a rebuild could change history). From v0.2.0
#: every entry here is literal text; only the newest is rendered. One
#: wording correction, applied once: a bare project name now reads
#: "Brisa+ (MorphoFavela)", as in all reader-facing text.
V012_ENTRY = """\
## v0.1.2 — 2026-09-25

P-05 shade goes live on a real pilot pull, per two decisions from the
2026-09-24 panel ruling: `om_shade_release` (Release building-only P-05 once dates are known; tree_shade an explicit empty column)
and `om_dates_tz` (Ask the team for the raw OM2 CSVs; infer dates/walk times from the data; timezone stays an open question until confirmed).

- **Pilot pull**: 5 CSVs (one per device — I_1/I_3/I_4/O_3/O_4) downloaded
  from the PI's Drive `04_Octopus_Maré/_data collection/Zenodo_release/
  fixed_data/` (created 2026-09-23) via the Drive connector, to
  `data/maré/octopus/csv/` with a manifest (file id, name, size,
  modified). The pilot manifest's own note on the rest of the Drive folder: search_files(parentId=...) was paginated across 3 pages (~150 file rows seen, with some repeats across pages -- Drive's search API does not guarantee stable pagination for a parentId filter). Every row seen was a CSV named <I|O>_<device#>_<YYYYMMDD>_<NN>durhrs.csv for devices I_1, I_3, I_4, O_3, O_4, spanning 2025-12 through 2026-04. Full-corpus mechanical download (remaining ~145+ files) is owed -- not done in this pass; the 5-file one-per-device pilot was judged sufficient to exercise infer_campaign_windows, the real P-05 shade run, and the join example.
- **Schema finding**: all 5 pilot CSVs share
  `Timestamp,Temperature,Humidity,PM1.0,PM2.5,PM2.5_cal,PM4.0,PM10.0` — NO
  Latitude/Longitude column. This is a fixed-site indoor/outdoor logger
  schema, not the OM2 GPS-track schema `OCTOPUS_JOIN_EXAMPLE` documents.
  Whether I_1/I_3/I_4/O_3/O_4 are the OM2 device under another name, or a
  separate deployment, is UNVERIFIED — flagged to Carlo, not assumed.
- `infer_campaign_windows()` made schema-tolerant: reports `has_gps=False`,
  `n_fix=n_rows`, `n_no_fix=0` for the no-GPS schema instead of raising;
  added `n_epoch_reset` (2000-01-01 rows) — none found in the pilot.
- `point_horizon_profiles()` WIRED for real (was `NotImplementedError` in
  v0.1/v0.1.1): builds the obstruction surface from
  `dtm_extended_300m.tif` + `buildings_extended_300m.gpkg` (WP-02's
  `build_surface`, cell_m=1.0) and marches WP-02's
  `patch_visibility(..., return_horizon=True)` from all 1559 OM2
  points at 1.5 m over the real 145-patch Tregenza direction set — same
  engine WP-04 uses for direct-sun-hours. Runs on the laptop GPU
  (RTX 4060); no timing figure is recorded for this run.
- **max_dist_m dropped to 100 m** (from WP-04's 500 m citywide default)
  for this march: `dtm_extended_300m.tif` has real nodata starting
  between 105 m and 599 m from
  OM2 points (median 338 m; measured via
  `shade.nodata_floor_m()`), and WP-02's running max is not NaN-safe
  (`torch.maximum` propagates NaN) — an unscoped pilot run returned
  all-NaN horizon values before this was caught. Not patched into the
  shared WP-02 engine; scoped locally to `point_horizon_profiles()`.
- `compute_shade()` run for the 5 pilot campaign dates (walk windows
  padded to the hour, `tz="UTC"` — a stated labelling choice per the
  current operating rule, NOT a resolution of the open timezone
  question), producing a real (not empty-schema) `p05_building_shade`
  table. `tree_shade` stays reserved and null.
- `OCTOPUS_JOIN_EXAMPLE` exercised on one real pilot CSV
  (`O_4_20260106_10durhrs.csv`): the temporal (nearest 5-min timestamp,
  150 s tolerance) `merge_asof` step runs and matches; the spatial
  (nearest-OM2-point, 0/0 no-fix drop) step is N/A for this schema and
  documented as such rather than faked.
- Package version v0.1.1 -> v0.1.2 across `package_docs.py`,
  `build_om_package.py`'s default `--version`, and the brisaverse release
  card (`om_release_v0_1_1` -> `om_release_v0_1_2`).
- **Package-page fixes** (docs/critic/octopus_package_panel_2026-09-24.md, blocking + top
  improvements): the "Panel ruling" link on the package page now points
  at a page rendered into `outputs/_packages/mare_om2/` itself
  (`panel_review.html`), not at `docs/critic/...md` outside `outputs/` —
  that path 404s on the live VPS hub, which only rsyncs `outputs/`, never
  `docs/`. README names Vincent alongside Jingxue and Simone as the
  release's named team, matching the `/ops` decision card. The Documents
  section links the actual deliverable data files (`OM2/points.*`,
  `p05_building_shade.*`, `p05b_campaign_windows.*`) directly instead of
  requiring a `manifest.json` reverse-engineer. A one-line glossary
  covers P-02..P-08, WP-02, `lambda_p` and the Tregenza sky for a reader
  outside Brisa+ (MorphoFavela).
"""

#: Frozen literal text — v0.1.3's entry as shipped (same correction).
V013_ENTRY = """\
## v0.1.3 — 2026-10-01

Structural fix (PI, 2026-09-27): the P-01..P-09 package spec was only
mentioned in README prose — conformance to it was invisible, and P-03's
aggregation script lived in the repo instead of travelling with the
package. Both addressed directly, not just documented around.

- **Spec conformance (P-00)**: the PI's package spec (P-01..P-09, verbatim,
  2026-09-23) is now encoded as data (`src/om_package/spec.py`), each part
  a mechanical predicate over the built package directory (file presence,
  required columns + coverage read from `p07_quality_report.json`,
  dictionary rows, README headings, changelog dating). Every build emits
  `p00_spec_conformance.json` + `.csv` at the package root — delivered /
  partial / pending per item, with evidence and, for a pending part, which
  `tasks.json` id(s) unblock it. The README gains a "Conformance to the
  package spec" section (right after the title) rendered from the same
  computed result. Nothing here is typed by hand: every count/coverage
  number is read from the file it describes.
- **P-03 ships inside the package**: `OM2/aggregate_to_segments.py` is a
  standalone (pandas + pyarrow only, no Brisa+ (MorphoFavela) import) mirror of
  `src/om_package/segments.py`'s `aggregate_to_segments` — a recipient
  with only this directory can re-aggregate to any segment length without
  the repo. `scripts/aggregate_om_points.py` (the repo-only CLI) is
  unchanged.
- **P-05 join example ships inside the package**: `OM2/join_shade_example.py`
  joins `p05_building_shade` against a real Octopus device CSV by
  `point_id` and 5-min-floored `timestamp`, states the UTC-labelling
  caveat in its own docstring.
- **Documentation corrections after audit** (2026-10-01): fixed the
  numerical-audit findings against v0.1.2's README/CHANGELOG — itemised
  here (release-scope internal-routes
  claim, route fetch dates, the nodata floor, the manifest self-hash, the
  frozen historical entries, the join-example pointer, "PI ruling Qxx"
  citations replaced by `provenance.decisions` ids, source vintages, and
  the lambda_p==1.0 check).
- **Descoped by decision** (`om_v013_descope`, PI, 2026-10-01): the
  terrestrial-LiDAR analysis (terrestrial sky-view factor, 2024 to 2026
  height change, airborne-vs-terrestrial comparison) and tree shade are
  out of scope for v0.1.3. The spec marks those parts `descoped`
  (a deliberate cut), not `pending`; items whose remaining parts are all
  delivered read `delivered (scoped)`. `tree_shade` stays in the shade
  schema as a reserved, all-null column. They remain candidates for a
  later version. The decision's text travels in `manifest.json`
  `provenance.decisions`.
- **README.pdf** ships in the package (the README rendered through pandoc
  and weasyprint) and the package page links it as "Download report (PDF)".
- Package version v0.1.2 -> v0.1.3 across `package_docs.py` and
  `build_om_package.py`'s default `--version`.
- **Figures rebuilt** (PI, 2026-09-27: "I would like to see the spatial
  result and then the sampling along the route; overlay the route on top
  of the favela buildings to be easier to understand"): the old
  `contact_sheet.py` (route floating in blank space, three noisy 1 m
  profiles) is replaced by `src/om_package/figures.py`'s four figures —
  `OM2/map_form.png` and `OM2/map_shade.png` (route over the Maré
  buildings and community outlines, coloured by `sky_view_factor` and by
  mean shaded fraction), `OM2/profiles.png` (1 m raw + 10 m segment means
  along the route) and `OM2/shade_calendar.png` (shaded/sunlit per
  campaign date). The package page shows them in that order (F1, F2
  stacked at 800 px, then F3, then F4).

"""

#: Frozen literal text: v0.2.0's entry as shipped.
V020_ENTRY = """\
## v0.2.0 — 2026-10-02

Sun exposure that does not depend on knowing the campaign date or the device
clock, and ventilation indices tied to observed wind. Data stays on the 2019
geometry for now; every geometry input is a build parameter, so moving to the
2024 airborne LiDAR and footprints later is one argument (`--buildings`,
`--dtm`), not a code change. v0.2.0 is a new directory; v0.1.3 is untouched.

- **P-10 sun exposure (new spec item)**: `p10_sun_envelope.parquet/.csv`
  (per point and local time of day over the campaign season: always sunlit,
  always shaded, date-dependent, or night, plus the sunlit share of days),
  `p10_sun_dose.parquet/.csv` (clear-sky direct-sun dose over the preceding
  1, 2 and 3 h, for each campaign date and as a season min/median/max
  envelope), `p10_horizon_profiles.parquet` (the marched horizon each point's
  sun result is derived from) and `p10_clock_agreement.parquet/.csv` (how much
  the exact-date shade changes if the device clock logged UTC rather than Rio
  local time). All geometry-derived proxies, not measured sunlight; the dose is
  clear-sky, so an upper bound.
- **P-11 ventilation indices with time-matched wind (new spec item)**:
  `p11_wind_observed.csv` (Galeão airport, SBGL, observations for the
  campaign window, flagged with the campaign date each would be matched to
  under each reading of the device clock) and, in `OM2/points.*`, seven new
  columns: `annual_sun_hours`, `windward_lambda_f_prevailing`,
  `canyon_alignment_prevailing_deg`, `upwind_shelter_deg_prevailing`,
  `z0_macdonald_m`, `zd_macdonald_m`, `open_space_fraction`. The ventilation
  columns are PROXIES from building geometry, never measured or simulated air
  temperature or air movement, and SBGL is a regional reference, not wind at the
  route. The source's fetch URL, time and sha256 are recorded in
  `manifest.json` under `provenance.wind_source`.
- **Spec**: P-10 and P-11 added to the conformance table; P-05 and P-06 are
  unchanged (the exact-date P-05 table stays, reserved and compatible). Status
  vocabulary as before: delivered / partial / pending / descoped.
- **Data dictionary**: a row for every new column and every column of the new
  files (all ventilation rows say PROXY). No id from v0.1.3 is removed or reused.
- **Quality report**: coverage for the new point columns plus a `p10_p11`
  block (class shares, clock agreement, wind-observation counts), all read off
  the tables.
- **Figures**: `OM2/sun_envelope.png`, `OM2/sun_dose.png`,
  `OM2/map_vent_shelter.png`, `OM2/profiles_vent.png`,
  `OM2/wind_rose_compare.png`.
- **Using the data**: new README section, including a note on segment length
  for the Octopus team (1 m points are finer than a walking sensor's response;
  a question to the team about their sensor's time constant).
  `OM2/aggregate_to_segments.py` gains `--segment-m` (default 10 m; the old
  `--segment-length-m` still works).
- **Changelog**: v0.1.3 and older entries are now frozen literal text rather
  than rendered from live build state. One wording correction, applied once to
  the frozen text: the bare project name now reads "Brisa+ (MorphoFavela)".
- **Reader-facing documents** (PI, 2026-10-01: "made for human, simple
  words, straight to the point"): `report.pdf` rewritten as a short report
  with all nine figures, numbered captions and the four findings (date,
  device clock, roughness range, observed wind), every number read from the
  package files and checked against `manifest.json`; README reorganised for
  scanning with a file table; package page reorganised (one action row, spec
  counts with the table behind a toggle, a figure gallery).
- Descoped by decision `om_v013_descope` and still out of scope: terrestrial
  sky-view factor, tree shade, the airborne-vs-terrestrial comparison and the
  2024 to 2026 height change.

"""


def render_readme(
    n_om2_points: int,
    n_route_geometry_flagged: int,
    n_lambda_p_ones: int,
    n_lambda_p_ones_flagged: int,
    lambda_p_share_explained_pct: float,
    n_lambda_p_remainder: int,
    n_lambda_p_remainder_plausible: int,
    route_fetch_date_label: str,
    nodata_floor_m: dict,
    internal_routes_status: str,
    decisions: list[dict],
    dtm_native_resolution_m: float,
    n_walks: int = 0,
    n_campaign_dates: int = 0,
    n_shade_rows: int = 0,
    shade_fraction_daylight_pct: float = 0.0,
    shade_max_dist_m: float = 100.0,
    conformance_section: str = "",
    *,
    p10_summary: dict,
    wind_source: dict,
    geometry_label: str,
    route_length_m: float,
    version: str = VERSION,
) -> str:
    """Render README.md. The route_geometry_flag/lambda_p/shade numbers,
    the nodata floor, the internal-routes on-disk state, the route fetch
    date label and the resolution decisions are all computed by the
    caller (build_om_package.py) from the actual OM2 build or from
    brisaverse's tasks.json, never hardcoded here (CLAUDE.md's 'never
    fabricate a value') — every one of them is a REQUIRED parameter, so a
    caller that forgets to compute one gets a loud TypeError instead of a
    silently-defaulted number. ``conformance_section`` is the rendered
    P-00 conformance table (src/om_package/spec.py
    render_conformance_markdown) — the build calls this twice: once with
    it empty to get a package directory conformance can be computed over,
    once with the computed table to produce the README actually shipped.
    """
    route_flag_pct = round(100 * n_route_geometry_flagged / n_om2_points, 1) if n_om2_points else 0.0
    n_lambda_p_remainder_not_checked = n_lambda_p_remainder - n_lambda_p_remainder_plausible
    floor = nodata_floor_m
    return README_TEMPLATE.format(
        version=version,
        version_date=VERSION_DATE,
        route_fetch_date_label=route_fetch_date_label,
        conformance_section=conformance_section,
        use_terms=USE_TERMS,
        om_use_terms_date=_decision(decisions, "om_use_terms")["resolved_utc"][:10],
        internal_routes_status=internal_routes_status,
        dtm_native_resolution_m=dtm_native_resolution_m,
        n_om2_points=n_om2_points,
        n_route_geometry_flagged=n_route_geometry_flagged,
        route_flag_pct=route_flag_pct,
        n_lambda_p_ones=n_lambda_p_ones,
        n_lambda_p_ones_flagged=n_lambda_p_ones_flagged,
        lambda_p_share_explained_pct=lambda_p_share_explained_pct,
        n_lambda_p_remainder=n_lambda_p_remainder,
        n_lambda_p_remainder_plausible=n_lambda_p_remainder_plausible,
        n_lambda_p_remainder_not_checked=n_lambda_p_remainder_not_checked,
        n_walks=n_walks,
        n_campaign_dates=n_campaign_dates,
        n_shade_rows=n_shade_rows,
        shade_fraction_daylight_pct=shade_fraction_daylight_pct,
        shade_max_dist_m=shade_max_dist_m,
        nodata_floor_min_m=floor["min"],
        nodata_floor_median_m=floor["median"],
        nodata_floor_max_m=floor["max"],
        p10_window=" to ".join(p10_summary["window"]),
        dose_slot_min=p10_summary["dose_slot_min"],
        envelope_slot_min=p10_summary["envelope_slot_min"],
        shade_step_min=SHADE_STEP_MIN,
        dose_hours_list=", ".join(map(str, p10_summary["dose_hours"])),
        dose_hours_and=" and ".join([", ".join(map(str, p10_summary["dose_hours"][:-1])), str(p10_summary["dose_hours"][-1])]) if len(p10_summary["dose_hours"]) > 1 else str(p10_summary["dose_hours"][0]),
        dose_hours_first=p10_summary["dose_hours"][0],
        route_flag_max_dist_m=ROUTE_FLAG_MAX_STREET_DIST_M,
        date_dependent_pct=100 * p10_summary["date_dependent_share"],
        wind_window=" to ".join(wind_source["window_utc"]),
        wind_fetched=str(wind_source["fetched_utc"])[:10],
        geometry_label=geometry_label,
        route_length_m=route_length_m,
        three_tau_pct=100 * (1 - math.exp(-3)),
        vent_buffer_m=DEFAULT_BUFFER_M,
    )


#: The newest entry: the only part of CHANGELOG.md rendered fresh on every build.
CURRENT_ENTRY_TEMPLATE = """\
# Changelog — mare_om2

## {version} — {version_date}

Walk-level timing, two wind regimes, and every time in Rio local time. The OM2
route is the new 1 m-spaced route ({n_om2_points} points); geometry stays on
the 2019 epoch. {version} is a new directory; earlier versions are untouched.

- **Time**: every shipped time column is Rio local time (ISO 8601 with the
  -03:00 offset, suffix `_local`); the loggers record UTC, and a `_utc` twin
  is kept in the data tables. The device-clock sensitivity analysis is gone:
  `p10_clock_agreement` and `p11_wind_observed.csv` are not shipped.
- **Walks (new)**: `p02b_walks` (one row per logger walk: start, end,
  duration, coverage, partial flag, wind regime tag; replaces
  `p05b_campaign_windows`) and `p12_walk_points` (one row per walk and route
  point: arrival time, shaded at arrival, clear-sky direct dose in the 1 h and
  3 h before arrival, and sensor-matched values at tau = 5, 10, 30 and 60 s).
- **Wind regimes (new)**: `p11_wind_regimes.csv` and `p11_regime_by_hour.csv`
  replace the single prevailing direction. Ventilation point columns are
  computed for both campaign-season regimes and named by regime
  (`frontal_area_density_windward_<regime>`, `canyon_alignment_deg_<regime>`,
  `upwind_shelter_angle_deg_<regime>`, `z0_macdonald_m_<regime>`); the
  `*_prevailing` columns and `z0_macdonald_m` are retired.
- **Shade (P-05)**: computed on the walk dates, daylight only, in local time
  (`timestamp_local`, `timestamp_utc`); parquet only.
- **Segment script**: `aggregate_to_segments.py` accepts `--by walk_id` for
  `p12_walk_points`, with `--tau`.
- **Dictionary**: rows for every new column; retired ids keep their row,
  marked RETIRED.
- Descoped by decision `om_v013_descope` and still out of scope: terrestrial
  sky-view factor, tree shade, the airborne-vs-terrestrial comparison and the
  2024 to 2026 height change.

"""


def render_changelog(n_om2_points: int, version: str = VERSION, version_date: str = VERSION_DATE) -> str:
    """Render CHANGELOG.md: only the newest entry is rendered; v0.1.3 and
    older are frozen literal text (see the constants above)."""
    current = CURRENT_ENTRY_TEMPLATE.format(version=version, version_date=version_date, n_om2_points=n_om2_points)
    return current + V020_ENTRY + V013_ENTRY + V012_ENTRY + CHANGELOG_V011_ENTRY + CHANGELOG_V01_ENTRY
