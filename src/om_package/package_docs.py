"""P-01 README + P-09 CHANGELOG for the mare_om2 package. Generated, not
hand-edited — rebuild via scripts/build_om_package.py after any source/method
change (never hand-edit data/outputs, per CLAUDE.md)."""
from __future__ import annotations

VERSION = "v0.1"
VERSION_DATE = "2026-09-24"

README_TEMPLATE = """\
# Maré morphology, OM2 — data package {version}

Built for Octopus LRP #2 ("Street by street: explaining air temperature
differences across streets and over time in Complexo da Maré", lead
Jingxue, PI Simone). Théo (PI, this repo) is a SUPPORT contributor:
street-form variables only. This package contains no temperature analysis
and no conclusions — that is the Octopus team's work.

Generated {version_date} by `scripts/build_om_package.py`
(source: `src/om_package/` in the MorphoFavela repo).

## CRS

SIRGAS 2000 / UTM 23S — EPSG:31983 — for every point, buffer and segment
geometry in this package. Route files arrive in WGS84 lon/lat (Google
Drive OM_1..OM_4_inferred_route.json) and are reprojected once, before
densification.

## Sources and dates

| Source | Date / vintage | Used for |
|---|---|---|
| OM_1..OM_4_inferred_route.json (Google Drive, PI-owned folder) | fetched {version_date} | P-02 route points |
| data/maré/raw/buildings_mare.shp + buildings_extended_300m.gpkg | 2019 airborne survey | P-04 building height/plan density, P-03 buffers |
| data/maré/raw/mare_dtm.tif (+ extended DTM) | 2019 | canyon H/W, SVF |
| data/maré/raw/street_mare.shp | 2019 | street network for SVF/canyon sampling |
| outputs/maré/svf_v2/svf_streets.gpkg | ray-cast from the above, 1.5 m pedestrian height, 145-patch Tregenza sky (src/svf_v2) | P-04 sky_view_factor |
| outputs/maré/morphometrics/canyon/hw_streets.gpkg | derived from the above (src/urban_morphology.py projected-width method) | P-04 street_width_m, building_height_m, height_width_ratio |
| outputs/maré/features/features_grid.parquet | 10 m grid, derived from the above | P-04 plan_density_lambda_p, P-06 ventilation proxies |
| data/maré/wind_rose.json | ASOS Galeão (SBGL) METAR, 2015-2024 | P-06 wind-alignment proxy |
| data/maré/neighbourhoods.gpkg | community boundary crosswalk | neighbourhood attribution |

**2019 buildings + terrain is the "airborne" source for every v0.1
variable.** No terrestrial (ground-instrument) source is used or
available yet — see Known limits.

## Methods

- **P-02 route points**: OM route edges are chained by `edge_order`,
  oriented geometrically (nearest-endpoint chaining — a stored LINESTRING
  may run opposite to its declared (u,v)), then sampled every 1 m along
  the chained centreline at 1.5 m pedestrian height (matches this repo's
  own SVF/street-sampling convention, src/svf_v2/sampling.py). No
  segments are imposed at this stage. Point IDs are deterministic:
  `<route>-<metres from route start, zero-padded>`, e.g. `OM2-000042`.
- **P-03 aggregation**: buffer variables (5/10/20/50 m circular buffers
  around each point) and segment aggregation (any length, on demand) are
  both re-runnable — `scripts/aggregate_om_points.py` for segments, buffer
  variables are computed at build time.
- **P-04 form variables**: nearest-neighbour spatial joins from the
  airborne sources above (each capped at a max join distance — beyond it
  a point gets NaN, never a guessed value) plus street_orientation_deg,
  computed directly from the route's own local tangent.
- **P-06 ventilation**: four PROXIES (never a flow simulation) — wind
  alignment, frontal-area density, openness, distance to open space. See
  the data dictionary (P-08) for exact formulas.
- **P-05 shade**: function + CLI (`src/om_package/shade.py`) that takes
  explicit campaign dates and a time window — campaign dates are not
  known yet, so v0.1 ships an empty-schema table, not a guessed demo run.

## Known limits

- **Airborne only.** Every P-04/P-06 variable is 2019 buildings+terrain
  derived. No terrestrial ground-truth comparison exists yet.
- **Terrestrial SVF: PENDING** — needs the team's 2026 OM2 terrestrial
  scan.
- **Building/tree shade: PENDING** — campaign dates unknown; no tree
  canopy/DSM layer for Maré on disk.
- **Height change 2024->2026: PENDING** — no re-survey on disk.
- Nearest-neighbour joins carry a `*_join_dist_m` column; check it before
  trusting a value near a data-layer edge.
- Ventilation columns are geometry-derived PROXIES, not simulated or
  measured airflow.
- om_routes.gpkg (Google Drive) was NOT fetched — too large for the
  connector. Pending if the team needs it.

## Use terms

PLACEHOLDER — the PI (Théo) to confirm before this package leaves the
project.

## How to cite

PLACEHOLDER — the PI (Théo) to confirm before this package leaves the
project.
"""

CHANGELOG_TEMPLATE = """\
# Changelog — mare_om2

## {version} — {version_date}

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
- P-08 data dictionary (every variable this package will ever carry,
  including PENDING rows).
- Contact sheet PNG for OM2.
"""


def render_readme() -> str:
    return README_TEMPLATE.format(version=VERSION, version_date=VERSION_DATE)


def render_changelog() -> str:
    return CHANGELOG_TEMPLATE.format(version=VERSION, version_date=VERSION_DATE)
