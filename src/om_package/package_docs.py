"""P-01 README + P-09 CHANGELOG for the mare_om2 package. Generated, not
hand-edited: rebuild via scripts/build_om_package.py.

render_readme(package_dir) reads every number from the built package
through report.compute_facts, the same facts the report uses, so the two
documents agree. The column list comes from p08_data_dictionary and the
files actually shipped.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from .vent_indices import DEFAULT_BUFFER_M

#: The one place the package version is set; the build default and every
#: rendered heading read it.
VERSION = "v0.3.1"
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
#: Who did what with the walk data; reused in the manifest, README and report.
DATA_CREDIT = (
    "The walks were collected by residents of Maré. Cassiano and Vincent (Octopus team) clean and structure "
    "the dataset."
)

USE_TERMS = "Internal draft for the Octopus team. Please do not share or cite."


def require_nodata_floor_m(p05_shade: dict) -> dict:
    """Pulls {"min", "median", "max"} nodata-floor stats (metres) out of a
    manifest's ``p05_shade`` block. Raises KeyError, never a default —
    this number used to be hardcoded (104.15 m) and drifted from what was
    actually measured; a manifest missing it means the build skipped the
    measurement (see shade.nodata_floor_m), and rendering must fail
    loudly rather than paper over that with a stale constant."""
    return p05_shade["nodata_floor_m"]


#: Spec item(s) per shipped file, for the README file table.
SPEC_ITEMS = {
    "OM2/points": "P-02, P-03, P-04, P-06, P-10, P-11",
    "p02b_walks": "P-12",
    "p12_walk_points": "P-12",
    "p05_building_shade": "P-05",
    "p10_sun_dose": "P-10",
    "p10_sun_envelope": "P-10",
    "p10_horizon_profiles": "P-10",
    "p11_wind_regimes": "P-11",
    "p11_regime_by_hour": "P-11",
    "p08_data_dictionary": "P-08",
    "OM2/p07_quality_report": "P-07",
    "OM2/aggregate_to_segments": "P-03",
    "OM2/join_shade_example": "P-05",
    "manifest.json": "",
}
#: Columns with no data dictionary row, described here.
_STRUCTURAL_COLUMNS = {
    "geometry": ("", "Point geometry (GeoParquet; the same point as `x`, `y`)."),
}
#: The quality report's own columns (CSV form; the JSON adds summary blocks).
_QUALITY_COLUMNS = {
    "variable": ("text", "Column of `OM2/points`."), "present": ("bool", "The column exists in the points table."),
    "coverage_fraction": ("fraction [0,1]", "Share of points with a value."),
    "n_valid": ("count", "Points with a value."), "n_total": ("count", "Points in the route."),
}
#: The data dictionary's own columns.
_DICTIONARY_COLUMNS = {
    "id": "Column name.", "definition": "What the column holds.", "unit": "Unit or type.",
    "source": "Input data.", "method": "How it is computed.", "limits": "What the value cannot tell.",
    "status": "computed, or why the column is empty or kept for compatibility.",
}


def _column_rows(package_dir: Path, f: dict) -> list[str]:
    """One table per data file, one row per column, from p08_data_dictionary.
    Sensor-matched columns (<measure>_tau<s>s) share one row per measure."""
    import re

    from .report import FILE_ORDER, _columns_of, _file_groups

    d = f["dictionary"]
    live = d[~d["status"].astype(str).str.startswith("RETIRED")].set_index("id")
    out: list[str] = []
    missing: list[str] = []
    groups = dict(_file_groups(package_dir))
    for stem in FILE_ORDER:
        exts = groups.get(stem)
        cols = _columns_of(package_dir, stem, exts) if exts else []
        if not cols:
            continue
        rows, seen = [], set()
        for c in cols:
            m = re.fullmatch(r"(.+)_tau(\d+)s", c)
            if m:
                base = m.group(1)
                if base in seen:
                    continue
                seen.add(base)
                taus = [re.fullmatch(rf"{re.escape(base)}_tau(\d+)s", x).group(1) for x in cols
                        if re.fullmatch(rf"{re.escape(base)}_tau(\d+)s", x)]
                name = f"`{base}_tau<{','.join(taus)}>s`"
                unit = str(live.loc[c, "unit"]) if c in live.index else ""
                definition = f"Sensor-matched `{base}` (see Methods, sensor-matched values)."
            elif stem == "p08_data_dictionary" and c in _DICTIONARY_COLUMNS:
                name, unit, definition = f"`{c}`", "text", _DICTIONARY_COLUMNS[c]
            elif stem == "OM2/p07_quality_report" and c in _QUALITY_COLUMNS:
                name, (unit, definition) = f"`{c}`", _QUALITY_COLUMNS[c]
            elif c in _STRUCTURAL_COLUMNS:
                name, (unit, definition) = f"`{c}`", _STRUCTURAL_COLUMNS[c]
            elif c in live.index:
                row = live.loc[c]
                name, unit = f"`{c}`", str(row["unit"])
                definition = ("Reserved column, always empty." if str(row["status"]).startswith("DESCOPED")
                              else " ".join(str(row["definition"]).split()))
            else:
                missing.append(f"{stem}:{c}")
                continue
            rows.append(f"| {name} | {unit.replace('|', '/')} | {definition.replace('|', '/')} |")
        label = stem if not exts[0] else f"{stem}.{'/'.join(sorted(exts, key=lambda e: e != 'parquet'))}"
        out += [f"### `{label}`\n", "| Column | Unit | Definition |", "|---|---|---|", *rows, ""]
    if missing:
        raise ValueError(f"columns with no row in p08_data_dictionary: {missing}")
    return out


def render_readme(package_dir) -> str:
    """README.md, every number from report.compute_facts over the built
    package, so the README and the report state the same values."""
    from pathlib import Path

    import pyproj

    from .report import (AUTHOR, PROJECT_FORM, _day, _join, _n, compute_facts, file_table,
                         opening_paragraph)
    from .sensor_match import DEFAULT_TAUS_S
    from .walk_dose import WALK_DOSE_HOURS, WALK_DOSE_STEP_MIN
    from .wind_regimes import N_SECTORS

    package_dir = Path(package_dir)
    f = compute_facts(package_dir)
    camp, clim = f["regimes"]["campaign"], f["regimes"]["climatology"]
    k1, k2 = sorted(camp)
    r1, r2 = camp[k1], camp[k2]
    y0, y1 = f["clim_years"]
    ws = f["wind_source"]
    floor = require_nodata_floor_m({"nodata_floor_m": f["nodata_floor_m"]})
    radii = sorted({int(c.split("_")[-1][:-1]) for c in
                    __import__("pyarrow.parquet", fromlist=["x"]).read_schema(package_dir / "OM2" / "points.parquet").names
                    if c.startswith("lambda_p_buffer_")})
    taus = _join([str(t) for t in DEFAULT_TAUS_S])
    walk_hours = _join([str(h) for h in WALK_DOSE_HOURS])
    p10_hours = _join([str(h) for h in f["p10_dose_hours"]])
    flag_pct = 100 * f["n_flagged"] / f["n_points"]
    z0_med = _join([f"{v:.3f} m" for v in f["z0_median"].values()])
    lines = [
        f"> **{USE_TERMS}**\n",
        "# Street form, sun and wind along the OM2 walking route, Complexo da Maré\n",
        f"{AUTHOR} · {PROJECT_FORM} · Octopus team · Octopus OM2 data package {f['version']}\n",
        opening_paragraph(f),
        "The report (`report.pdf`) presents each measure with figures. This README gives the method, the "
        "sources and every column.\n",
        "## Files in this package\n",
        file_table(package_dir, f, spec_items=SPEC_ITEMS),
        "Also shipped: `README.md` and `README.pdf` (this document, spec item P-01), `report.md` and "
        "`report.pdf` (the report) and `OM2/fig_*.png` (the report figures).\n",
        "## Sources and dates\n",
        "| Source | Date | Used for |",
        "|---|---|---|",
        f"| Walk dataset: walks collected by residents of Maré, cleaned and structured by Cassiano and Vincent (Octopus team); matched GPS tracks, one CSV per walk, and the OM2 "
        f"route file | {f['n_walks']} walks on {f['n_dates']} dates, {_day(f['first_date'])} to "
        f"{_day(f['last_date'])}; the dataset's pre-release manifest records a SHA-256 checksum per file | route "
        "points, walk timing, arrival times, walk dates |",
        f"| Building footprints with heights and the terrain model | {f['geometry_epoch']} | street form, shade, "
        "sun, ventilation, `route_geometry_flag` |",
        "| Street centre lines of Maré | same layer set | street form sampling, `route_geometry_flag` |",
        f"| Galeão airport hourly weather reports (Iowa Environmental Mesonet archive) | campaign season "
        f"{_day(ws['window_utc'][0])} to {_day(ws['window_utc'][1])}, fetched {_day(f['wind_fetched'])}; "
        f"{y0} to {y1} for the long-term regimes | wind regimes, walk wind tags |",
        "| Neighbourhood boundaries of Maré | | `neighbourhood` |",
        "",
        "The fetch address and SHA-256 checksum of the airport reports are in `manifest.json` under "
        "`provenance.wind_source`.\n",
        "## CRS (coordinate reference system)\n",
        f"{pyproj.CRS(f['crs']).name} ({f['crs']}) for every point and geometry. The `x` and `y` columns are in "
        "metres in this system.\n",
        "## Methods\n",
        "### Route points\n",
        f"The route is the walk dataset's OM2 route: {_n(f['length_m'])} m, sampled every {f['spacing_m']:g} m "
        f"along its centre line into {_n(f['n_points'])} points at {f['height_m']:g} m above the ground. "
        "`point_id` is `OM2-` plus the distance from the route start in metres, zero-padded. Some points fall "
        "inside building outlines or far from a street centre line, because some alleys cannot be mapped; "
        f"`route_geometry_flag` marks the {_n(f['n_flagged'])} points ({flag_pct:.1f}%) that lie inside a "
        f"building outline or more than {f['flag_dist_m']:g} m from a street centre line.\n",
        "### Street form\n",
        "Building height, street width and their ratio come from cross-sections of the flanking buildings at "
        "street samples; sky view factor is ray-cast from 1.5 m above street samples over 145 sky patches "
        "against the 2019 buildings and terrain; plan density is the building share of the 10 m grid cell. Each "
        "point takes the nearest sample within a maximum distance and stays empty beyond it, never a guessed "
        "value; `*_join_dist_m` gives that distance. The height-to-width ratio is computed per point; the "
        f"median of the point ratios is {f['hw_median_of_ratios']:.2f}, and the ratio of the median height to "
        f"the median width is {f['hw_ratio_of_medians']:.2f}. Buffer columns give plan density, building count "
        f"and mean building height within {_join([str(r) for r in radii])} m of each point.\n",
        "### Shade and sun\n",
        "Sun, shade and ventilation are computed from 2019 building and terrain geometry. For each point the "
        f"horizon (the angle of the highest building or terrain) is marched once in every direction up to "
        f"{f['shade_max_dist_m']:g} m, below the distance where the terrain data first run out "
        f"({floor['min']:.0f} m at the nearest point, {floor['median']:.0f} m at the median point). A point is in "
        "building shade when the sun is below that horizon. The shade table covers every "
        f"{f['shade_step_min']} minutes of daylight on the {f['n_dates']} walk dates, in Rio local time "
        f"(`timestamp_local`) with a UTC twin (`timestamp_utc`); the route is in building shade for "
        f"{100 * f['shade_daylight']:.1f}% of daylight time.\n",
        "The **direct sun dose** is the clear-sky direct beam energy on a horizontal surface, in Wh/m². It "
        "assumes a clear sky, so it is an upper bound. `p12_walk_points` gives it for the "
        f"{walk_hours} hours before each walk reached each point, summed in {WALK_DOSE_STEP_MIN}-minute steps "
        f"from the walk's GPS arrival times. `p10_sun_dose` gives it for the past {p10_hours} hours at "
        f"{f['p10_dose_slot_min']}-minute times of day, for each walk date and as the lowest, median and highest "
        f"over the season {_day(f['p10_window'][0])} to {_day(f['p10_window'][1])}. `p10_sun_envelope` classes "
        f"each point and {f['p10_envelope_slot_min']}-minute time of day over that season as always sunlit, "
        "always shaded or sunlit on some dates only. `annual_sun_hours` counts the hours per year with the sun "
        "above the point's horizon.\n",
        "### Walk timing and arrival times\n",
        "Only GPS fixes matched to edges of the OM2 route count. Walkers only move forward, so the distance "
        "along the route is a running maximum over time. A point's arrival time is interpolated in time along "
        "the route between the fixes before and after it; where the walker stopped, it is the first moment the "
        f"point was reached. `arrival_source` is `gap_interpolated` when those fixes are more than "
        f"{f['gap_flag_s']} s apart ({100 * f['gap_share']:.1f}% of walk points) and `gps` otherwise. A walk "
        f"whose fixes cover less than {100 * f['partial_coverage']:.0f}% of the route is marked `partial` "
        f"({f['n_partial']} of {f['n_walks']} walks). Points a walk did not reach have no row.\n",
        "### Wind regimes\n",
        "Reports with a speed below the calm threshold "
        f"({ws['calm_threshold_ms']:g} m/s) or no direction are set aside. The primary method takes the "
        f"{N_SECTORS}-sector wind rose, smooths it with weights 1, 2, 1 over neighbouring sectors and keeps the "
        "two highest peaks at least two sectors apart; each report goes to the nearer peak, and a regime's "
        "direction is the circular mean of its reports. The check fits two von Mises distributions plus a "
        "uniform background to the directions (with a uniform spread of 5° to undo the 10° reporting steps). In "
        f"the campaign season the check confirms the {r1['name']} regime (nearest fitted direction "
        f"{r1['mix_dir']:.0f}°, {r1['mix_diff']:.0f}° from the regime) but not the {r2['name']} one (nearest "
        f"fitted direction {r2['mix_dir']:.0f}°, {r2['mix_diff']:.0f}° away): the {r2['name']} reports spread "
        f"over a broad northern arc. {y0} to {y1} gives the same picture. Each walk takes the regime of the "
        f"report nearest its middle time, if that report is within {f['tag_max_gap_min']} minutes and has a "
        "direction.\n",
        "| Period | Regime | Mean direction | Share of reports | Mean speed |",
        "|---|---|---|---|---|",
        *[f"| {per} | {g['name']} | {g['dir']:.0f}° | {100 * g['share']:.1f}% | {g['speed']:.1f} m/s |"
          for per, regs in (("campaign season", camp), (f"{y0} to {y1}", clim)) for g in regs.values()],
        "",
        "### Ventilation measures\n",
        "Each measure is computed at the mean direction of each campaign-season regime, and its columns end "
        f"in the regime name (`_{r1['slug']}`, `_{r2['slug']}`). None is a measured or simulated wind. Frontal "
        "area density facing the wind is interpolated between the eight compass columns of the 10 m grid cell. "
        "Canyon alignment is the angle between the street axis and the wind, from 0° (along) to 90° (across). "
        "Upwind shelter angle is the horizon angle in the direction the wind comes from. Open space fraction "
        f"is the unbuilt share of the {DEFAULT_BUFFER_M} m buffer.\n",
        "Roughness length and displacement height follow Macdonald et al. (1998), from the plan density and "
        f"mean building height of the {DEFAULT_BUFFER_M} m buffer and the frontal area density facing the wind. "
        "The method was calibrated on regular arrays of blocks, sparser than Maré. Along most of the route the "
        f"displacement height approaches the roof height (median {f['zd_median']:.1f} m) and the roughness "
        f"length falls towards zero (medians {z0_med}): read these values as outside the calibrated range. "
        "They are in the data only.\n",
        "### Sensor-matched values\n",
        "A sensor carried along the route reads air it has already passed. The sensor-matched value of a "
        "measure X at point i is the weighted mean of X over the points j the walk had passed, with weights "
        "exp(-Δt/τ), Δt = t_i - t_j from the walk's arrival times, leaving out points more than "
        f"{f['truncation_taus']:g}τ back and scaling the weights to sum to one. Empty X values are skipped. τ is "
        "the sensor time constant, the time to reach 63% of a step change; if only the 90% response time t90 "
        f"is known, τ = t90 / {f['ln10']:.3f}. Columns are given for τ = {taus} s.\n",
        "### Segment script\n",
        "`OM2/aggregate_to_segments.py` (pandas and pyarrow only) averages points over segments of any "
        "length. Run it from inside the package directory:\n",
        "```\npython OM2/aggregate_to_segments.py --points OM2/points.parquet \\\n"
        "    --segment-m 20 --out OM2/segments_20m.parquet\n```\n",
        "For one row per walk and segment, with one time constant:\n",
        "```\npython OM2/aggregate_to_segments.py --points p12_walk_points.parquet \\\n"
        "    --by walk_id --segment-m 20 --tau 30 --out segments_by_walk.parquet\n```\n",
        "A segment ends at the point a reading was taken; a sensor reading describes the route behind the walker.\n",
        "## Using the data\n",
        "Loggers record UTC; Rio local time is UTC-3 with no daylight saving. Join logger readings to "
        "`p12_walk_points` by `walk_id` and the nearest `t_arrival_utc`, or to the shade table by `point_id` "
        "and `timestamp_utc` floored to the shade step (`OM2/join_shade_example.py`). Run each analysis with "
        "and without the points flagged by `route_geometry_flag`, and down-weight or drop rows whose "
        "`arrival_source` is `gap_interpolated`.\n",
        "## Known limits\n",
        "- Street form, sun, shade and ventilation come from 2019 building and terrain geometry.\n"
        "- The sun dose assumes a clear sky, so it is an upper bound.\n"
        "- The airport wind is a regional reference, not the wind in the streets.\n"
        "- An empty value in a joined column means no source sample within the join distance; an empty buffer "
        "mean means no building in the buffer.\n",
        "## Columns\n",
        *_column_rows(package_dir, f),
        "## Manifest\n",
        "`manifest.json` records the package version, the coordinate system, the use terms, the decisions "
        "behind this release, the wind source, summary values and a SHA-256 checksum for every other file. "
        "It excludes its own hash, because a file cannot record its own checksum.\n",
        "## Use terms\n",
        f"{USE_TERMS}\n",
        "## How to cite\n",
        f"Please do not cite this draft. The package was produced with the {PROJECT_FORM} pipeline by {AUTHOR}; "
        "authorship is to be discussed with the lead author when the contribution list is drafted.\n",
        "## Contact\n",
        f"{AUTHOR}, {PROJECT_FORM}.\n",
    ]
    return "\n".join(lines)


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
