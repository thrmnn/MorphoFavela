"""P-01 README + P-09 CHANGELOG for the mare_om2 package. Generated, not
hand-edited — rebuild via scripts/build_om_package.py after any source/method
change (never hand-edit data/outputs, per CLAUDE.md)."""
from __future__ import annotations

VERSION = "v0.1.2"
VERSION_DATE = "2026-09-25"

#: PI ruling 2026-09-24 (Q6a). Same text goes in the README banner and the
#: manifest's use_terms field — one string, so the two can never drift.
USE_TERMS = (
    "INTERNAL REVIEW DRAFT — Octopus LRP #2 team members only; not for "
    "redistribution or citation; to be revisited at submission."
)

README_TEMPLATE = """\
> **{use_terms}**

# Maré morphology, OM2 — data package {version}

Built for Octopus LRP #2 ("Street by street: explaining air temperature
differences across streets and over time in Complexo da Maré", lead
Jingxue, PI Simone). Théo Hermann contributes street-form variables from
the MorphoFavela pipeline in a support role. This package contains no
temperature analysis and no conclusions — those are the Octopus team's work.

**Named team (PI ruling 2026-09-24, Q6a):** Jingxue, Vincent, Simone —
this release goes to the three of them only, as an internal review draft;
not for wider redistribution or citation (see Use terms).

Generated {version_date} by `scripts/build_om_package.py`
(source: `src/om_package/` in the MorphoFavela repo).

## Release scope

**OM2 only.** This directory (`outputs/_packages/mare_om2/{version}/`)
contains OM2 exclusively. OM1/OM3/OM4 are built by the exact same code
path (`--route ALL`) but written to an internal build directory outside
this package (`outputs/_packages/_internal/mare_routes/{version}/`) — they
are never copied or symlinked into `mare_om2/`, so zipping/sharing this
directory cannot leak them. **{version} supports RQ2's within-route
spatial/temporal analysis only.** RQ3's between-route holdout needs
OM1/OM3/OM4 released, which is a PI decision for a later version.

## Coverage vs Table 1

**This package covers surface structure only** — route-point geometry,
building height/plan density/canyon form, ventilation-geometry proxies,
and (once dates are known) building shade. Surface cover (vegetation
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
| OM_1..OM_4_inferred_route.json (Google Drive, PI-owned folder) | fetched {version_date} | P-02 route points |
| data/maré/raw/buildings_mare.shp + buildings_extended_300m.gpkg | 2019 airborne survey | P-04 building height/plan density, P-03 buffers, route_geometry_flag |
| data/maré/raw/mare_dtm.tif (+ extended DTM) | 2019 | canyon H/W, SVF |
| data/maré/raw/street_mare.shp | 2019 | street network for SVF/canyon sampling, route_geometry_flag |
| outputs/maré/svf_v2/svf_streets.gpkg | ray-cast from the above, 1.5 m pedestrian height, 145-patch Tregenza sky (src/svf_v2) | P-04 sky_view_factor |
| outputs/maré/morphometrics/canyon/hw_streets.gpkg | derived from the above (src/urban_morphology.py projected-width method) | P-04 street_width_m, building_height_m, height_width_ratio |
| outputs/maré/features/features_grid.parquet | 10 m grid, derived from the above | P-04 plan_density_lambda_p, grid_cell_id, P-06 ventilation proxies incl. lambda_f_<dir> |
| data/maré/wind_rose.json | ASOS Galeão (SBGL) METAR, 2015-2024 | P-06 wind-alignment proxy |
| data/maré/neighbourhoods.gpkg | community boundary crosswalk | neighbourhood attribution |

**2019 buildings + terrain is the "airborne" source for every {version}
variable.** No terrestrial (ground-instrument) source is used or
available yet — see Known limits.

## Methods

- **P-02 route points**: OM route edges are chained by `edge_order`,
  oriented geometrically (nearest-endpoint chaining — a stored LINESTRING
  may run opposite to its declared (u,v)), then sampled every 1 m along
  the chained centreline at 1.5 m pedestrian height (matches this repo's
  own SVF/street-sampling convention, src/svf_v2/sampling.py). No
  segments are imposed at this stage. Point IDs are deterministic:
  `<route>-<metres from route start, zero-padded>`, e.g. `OM2-000042` —
  and PROVISIONAL (see Known limits).
- **P-03 aggregation**: buffer variables (5/10/20/50 m circular buffers
  around each point) and segment aggregation (any length, on demand) are
  both re-runnable — `scripts/aggregate_om_points.py` for segments, buffer
  variables are computed at build time.
- **P-04 form variables**: nearest-neighbour spatial joins from the
  airborne sources above (each capped at a max join distance — beyond it
  a point gets NaN, never a guessed value) plus street_orientation_deg,
  computed directly from the route's own local tangent, plus
  `grid_cell_id` (the same features_grid cell used for
  `plan_density_lambda_p`) so models can cluster the 10 m-grid variables.
- **P-06 ventilation**: four PROXIES (never a flow simulation) — wind
  alignment, OMNIDIRECTIONAL frontal-area density, openness, distance to
  open space — plus the 8 per-compass-direction frontal-area columns
  (`lambda_f_N` .. `lambda_f_NW`) passed through unchanged, so a
  windward-specific figure can be built downstream without this package
  guessing the wind direction that matters. See the data dictionary
  (P-08) for exact formulas.
- **route_geometry_flag**: True where a point falls inside a
  `buildings_mare` footprint OR more than 10 m from the nearest
  `street_mare` centreline (`src/om_package/routes.py`,
  `ROUTE_FLAG_MAX_STREET_DIST_M`) — both are signs the OSM-inferred route
  drifted off the street the team actually walked. See Known limits for
  the measured counts.
- **P-05 shade**: function + CLI (`src/om_package/shade.py`), now RUN for
  real against a {n_csv_pilot}-file pilot pull from the team's Drive
  (`Zenodo_release/fixed_data/`, one CSV per device: I_1/I_3/I_4/O_3/O_4).
  `infer_campaign_windows()` read {n_campaign_dates} campaign dates/walk
  windows straight off those files (no epoch-reset rows found in the
  pilot). `point_horizon_profiles()` is now WIRED to the real WP-02/WP-04
  horizon engine (`src.brisa_solar.wp02_surface.build_surface` +
  `wp02_horizon.patch_visibility(..., return_horizon=True)`, real 145-patch
  Tregenza directions) and was run for all {n_om2_points} OM2 points on the
  laptop GPU. `compute_shade()` then produced {n_shade_rows} real
  (point x 5-min-timestamp) rows across the {n_campaign_dates} campaign
  dates (walk windows padded to the hour), **computed in UTC**
  ({shade_fraction_pct}% of rows shaded) — see Known limits for why UTC
  and why `max_dist_m={shade_max_dist_m:g} m`, not WP-04's 500 m citywide
  default. The schema reserves a `tree_shade` column (always null).

## Known limits

- **Airborne only.** Every P-04/P-06 variable is 2019 buildings+terrain
  derived. No terrestrial ground-truth comparison exists yet.
- **Sky-view factor is an UPPER BOUND under canopy.** The ray-cast mesh is
  buildings + bare-earth terrain only — no vegetation is in the scene — so
  a tree-covered point's real sky view is <= the reported
  `sky_view_factor`, never more.
- **route_geometry_flag**: {n_route_geometry_flagged}/{n_om2_points}
  OM2 points ({route_flag_pct}%) are flagged — inside a building footprint
  or more than 10 m from the nearest street centreline. Of the
  {n_lambda_p_ones} points with `plan_density_lambda_p == 1.0`,
  {n_lambda_p_ones_flagged} ({lambda_p_share_explained_pct}%) are
  explained by this flag; the remainder are plausible fully-built 10 m
  cells, not a join defect.
- **point_id is PROVISIONAL.** It is minted from the OSM-inferred route
  file, not the team's own om_routes.gpkg. The ID string is stable across
  rebuilds of the same route file, but the place it names may move when
  v0.2 rebuilds on the real route — that release will publish an
  old->new `point_id` crosswalk.
- **Timezone is still UNRESOLVED** for the OM2 campaign: GPS-fix Timestamp
  rows are UTC per firmware; RTC-fallback (no-fix) rows may be local time
  or something else the firmware does not record. Every function in
  `src/om_package/shade.py` that needs a timezone takes it as a required
  parameter with no default. **{version}'s P-05 table is computed with
  `tz="UTC"`** (current operating rule for this cycle) — this is a stated
  labelling choice, NOT a resolution of the open question; treat every
  timestamp in `p05_building_shade` as UTC-labelled, re-derive if the team
  confirms otherwise, and do not read it as local Rio clock time.
- **The Zenodo_release/fixed_data pilot CSVs have NO Latitude/Longitude
  column** (`Timestamp,Temperature,Humidity,PM1.0,PM2.5,PM2.5_cal,PM4.0,
  PM10.0` — confirmed on one CSV per device, I_1/I_3/I_4/O_3/O_4). This
  looks like a fixed-site indoor/outdoor logger schema, not the OM2
  GPS-track schema `OCTOPUS_JOIN_EXAMPLE` documents (which needs
  Latitude/Longitude to pick the nearest OM2 point and to drop 0/0 no-fix
  rows). Whether I_1/I_3/I_4/O_3/O_4 ARE the OM2 device under another
  naming convention, or a separate fixed-site deployment, is UNVERIFIED —
  see `docs/research/octopus_lidar_sources.md` §5 and the team message.
  `infer_campaign_windows()` was made schema-tolerant (v0.1.2): it reports
  `has_gps=False`, `n_fix=n_rows`, `n_no_fix=0` for this schema rather
  than raising. The join example below is exercised on a real file from
  this pull with its temporal (nearest 5-min timestamp) step only, since
  the spatial (nearest-OM2-point) step needs a GPS-track CSV this pull
  did not contain.
- **max_dist_m={shade_max_dist_m:g} m, not WP-04's 500 m citywide
  default**, for the horizon march behind P-05: `dtm_extended_300m.tif` /
  `buildings_extended_300m.gpkg` has real nodata starting {shade_min_nan_dist_m:.0f}
  m from the nearest OM2 point (measured 2026-09-25 — its raster bounding
  box is a rectangle, but valid coverage inside it is not). WP-02's
  horizon march (`wp02_horizon.py`) is not NaN-safe (`torch.maximum`
  propagates NaN), so a full-radius pilot run returned all-NaN horizon
  values before this was caught; {shade_max_dist_m:g} m is safely under
  every OM2 point's measured nodata floor and was NOT patched into the
  shared WP-02 engine (P1's citywide/WP-04 defended numbers also depend on
  it) — this scoping fix lives only in `point_horizon_profiles()`.
- **NaN has two distinct causes** in the point table's joined columns —
  they are not interchangeable and are documented separately per column
  in the data dictionary (P-08): (1) *beyond the join-distance cap*
  (`building_height_m`, `sky_view_factor`, `plan_density_lambda_p`,
  `grid_cell_id`, ventilation proxies — a point too far from any source
  sample); (2) *no feature in the buffer* (`building_height_mean_buffer_*m`
  — zero buildings intersect that point's buffer, a real "no building
  here" result, not a join gap).
- **Terrestrial SVF: PENDING** — needs the team's 2026 OM2 terrestrial
  scan.
- **Building shade: computed for {n_campaign_dates} pilot campaign dates**
  (see P-05 above) — more dates arrive as more of the team's Drive CSVs
  are pulled; an empty table still ships when no CSVs are found at build
  time. **Tree shade: PENDING** — no tree canopy/DSM layer for Maré on
  disk. `tree_shade` is reserved as an always-null column in the shade
  schema.
- **Height change 2024->2026: PENDING** — data location being confirmed
  by T. Hermann.
- Nearest-neighbour joins carry a `*_join_dist_m` column; check it before
  trusting a value near a data-layer edge.
- Ventilation columns are geometry-derived PROXIES, not simulated or
  measured airflow; they are isotropic, so they say nothing about upwind
  fetch beyond axis alignment.
- om_routes.gpkg (Google Drive) was NOT fetched — too large for the
  connector. Pending if the team needs it.

## Manifest

`manifest.json` records `package_version`, `crs`, `use_terms`, per-route
build stats (relative output paths), and a `sha256` per file in this
package (recompute and compare before trusting a copy). Parquet tables
built from a GeoDataFrame (`points.parquet`) carry GeoParquet `geo`
metadata as well as plain `x`/`y` columns, so both GeoParquet-aware and
plain-pandas readers work without extra steps.

## Use terms

{use_terms}

## How to cite

This package was produced with the MorphoFavela pipeline (Théo Hermann).
Authorship is to be discussed with the lead author when the Octopus LRP #2
contribution list is drafted.
"""

CHANGELOG_TEMPLATE = """\
# Changelog — mare_om2

## {version} — {version_date}

P-05 shade goes live on a real pilot pull, per the PI ruling 2026-09-24
(Q1/Q5: release building-only shade once dates are known, from the raw
CSVs, timezone stays UNRESOLVED).

- **Pilot pull**: 5 CSVs (one per device — I_1/I_3/I_4/O_3/O_4) downloaded
  from the PI's Drive `04_Octopus_Maré/_data collection/Zenodo_release/
  fixed_data/` (created 2026-09-23) via the Drive connector, to
  `data/maré/octopus/csv/` with a manifest (file id, name, size,
  modified). The full folder holds far more files than this pilot pulled
  (catalogued, not all downloaded — see the manifest's `catalogued_not_downloaded`
  count and `docs/research/octopus_lidar_sources.md` §5).
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
  `patch_visibility(..., return_horizon=True)` from all 1559 OM2 points
  at 1.5 m over the real 145-patch Tregenza direction set — same engine
  WP-04 uses for direct-sun-hours. Runs on the laptop GPU (RTX 4060),
  ~9 s for all 1559 points.
- **max_dist_m dropped to 100 m** (from WP-04's 500 m citywide default)
  for this march: `dtm_extended_300m.tif` has real nodata starting
  ~104-330 m from OM2 points, and WP-02's running max is not NaN-safe
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
- **Package-page fixes** (navigation council panel review 2026-09-24,
  blocking + top improvements): the "Panel ruling" link on the package
  page now points at a page rendered into `outputs/_packages/mare_om2/`
  itself (`panel_review.html`), not at `docs/critic/...md` outside
  `outputs/` — that path 404s on the live VPS hub, which only rsyncs
  `outputs/`, never `docs/`. README names Vincent alongside Jingxue and
  Simone as the release's named team, matching the `/ops` decision card.
  The Documents section links the actual deliverable data files
  (`OM2/points.*`, `p05_building_shade.*`, `p05b_campaign_windows.*`)
  directly instead of requiring a `manifest.json` reverse-engineer. A
  one-line glossary covers P-02..P-08, WP-02, `lambda_p` and the Tregenza
  sky for a reader outside MorphoFavela.

## v0.1.1 — 2026-09-24

Panel review (docs/critic/octopus_package_panel_2026-09-24.md) and PI
ruling (interview 2026-09-24) applied on top of v0.1's initial release.

- Release scope narrowed to **OM2 only** in the shared package path; OM1/
  OM3/OM4 now build to `outputs/_packages/_internal/mare_routes/{version}/`,
  never inside `mare_om2/`.
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
- How to cite: filled with an acknowledgment line (MorphoFavela pipeline,
  Théo Hermann); authorship to be raised with the lead author when the
  contribution list is drafted. No PLACEHOLDER remains anywhere in the
  README.

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


def render_readme(
    n_om2_points: int,
    n_route_geometry_flagged: int,
    n_lambda_p_ones: int,
    n_lambda_p_ones_flagged: int,
    lambda_p_share_explained_pct: float,
    n_csv_pilot: int = 0,
    n_campaign_dates: int = 0,
    n_shade_rows: int = 0,
    shade_fraction_pct: float = 0.0,
    shade_max_dist_m: float = 100.0,
    shade_min_nan_dist_m: float = 104.15,
) -> str:
    """Render README.md. The route_geometry_flag/lambda_p/shade numbers are
    computed by the caller (build_om_package.py) from the actual OM2
    build, never hardcoded here — see CLAUDE.md's 'never fabricate a
    value'."""
    route_flag_pct = round(100 * n_route_geometry_flagged / n_om2_points, 1) if n_om2_points else 0.0
    return README_TEMPLATE.format(
        version=VERSION,
        version_date=VERSION_DATE,
        use_terms=USE_TERMS,
        n_om2_points=n_om2_points,
        n_route_geometry_flagged=n_route_geometry_flagged,
        route_flag_pct=route_flag_pct,
        n_lambda_p_ones=n_lambda_p_ones,
        n_lambda_p_ones_flagged=n_lambda_p_ones_flagged,
        lambda_p_share_explained_pct=lambda_p_share_explained_pct,
        n_csv_pilot=n_csv_pilot,
        n_campaign_dates=n_campaign_dates,
        n_shade_rows=n_shade_rows,
        shade_fraction_pct=shade_fraction_pct,
        shade_max_dist_m=shade_max_dist_m,
        shade_min_nan_dist_m=shade_min_nan_dist_m,
    )


def render_changelog() -> str:
    return CHANGELOG_TEMPLATE.format(version=VERSION, version_date=VERSION_DATE)
