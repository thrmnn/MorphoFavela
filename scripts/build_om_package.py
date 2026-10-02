#!/usr/bin/env python3
"""Build the "Maré morphology, OM2" data package — Octopus LRP #2
("Street by street: explaining air temperature differences across streets
and over time in Complexo da Maré", lead Jingxue, PI Simone). Théo (PI) is
a SUPPORT contributor here, supplying street-form variables only — this
script and everything under src/om_package/ never compute or state a
temperature conclusion; that is the Octopus team's analysis, not ours.

Release scope (PI ruling 2026-09-24): the SHARED package path
(outputs/_packages/mare_om2/<version>/) contains OM2 only. OM1/OM3/OM4
share the exact same code path (pass --route ALL) but are written to an
INTERNAL build directory (outputs/_packages/_internal/mare_routes/<version>/)
that is never copied into the shared package.

Builds, per requested route:
  P-02 route points (1 m spacing, pedestrian height, stable IDs) +
       route_geometry_flag (within a building OR >10 m from the nearest
       street centreline — src/om_package/routes.py)
  P-03 buffer variables (5/10/20/50 m) — segment aggregation is a
       separate script, scripts/aggregate_om_points.py, since the segment
       length is the team's choice, not fixed at build time
  P-04 airborne form variables (incl. grid_cell_id)
  P-06 ventilation proxies (incl. the 8 lambda_f_<dir> columns)
  P-05 shade — walk dates, daylight only, Rio local time (see
       src/om_package/shade.py) — OM2/shared package only
  P-10 sun exposure over the season: envelope, dose, horizon profiles,
       annual_sun_hours
  P-11 two SBGL wind regimes (p11_wind_regimes, p11_regime_by_hour) and the
       ventilation point columns at each regime (all PROXIES from building
       geometry)
  P-12 walks (p02b_walks) and per-walk point values (p12_walk_points)
  P-07 quality report (counts route_geometry_flag)
  P-08 data dictionary — OM2/shared package only
  contact sheet PNG (OM2 only)
  neighbourhoods crossed (printed + saved)
  manifest.json: package_version, crs, use_terms, relative paths, sha256
       per file (computed last, over every file this run wrote)
  outputs/_packages/mare_om2/index.html: the package page (stable URL
       across versions), rebuilt from this run's outputs by
       scripts/build_om_package_page.py — see that module for its content.

Run:
    python scripts/build_om_package.py --route OM2 --root /home/theo/SCL/SCR/MorphoFavela
    python scripts/build_om_package.py --route ALL --root /home/theo/SCL/SCR/MorphoFavela

Geometry epoch (2019 now; 2024 ALS + footprints later): every geometry input
is a path, so swapping the epoch is --buildings / --dtm (+ --geometry-epoch
label) — nothing in src/om_package/ hardcodes the vintage.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for build_om_package_page,
# a sibling module — needed when this file is loaded via importlib (tests)
# rather than run as `python scripts/build_om_package.py`.

import geopandas as gpd
import pandas as pd

from src.om_package.buffers import BUFFER_RADII_M, compute_buffer_variables
from src.om_package.dictionary import dictionary_dataframe
from src.om_package.figures import (
    build_fig_form,
    load_shade_frame,
    build_fig_route,
    build_fig_shade_calendar,
    build_fig_shade_map,
    build_fig_sun_dose,
    build_fig_svf_sensor,
)
from src.om_package.vent_figures import build_fig_shelter_maps, build_fig_vent_profiles, build_fig_wind
from src.om_package.formvars import compute_form_variables
from src.om_package.io_utils import Paths, hash_tree, write_table
from src.om_package.neighbourhoods import communities_crossed, join_communities
from src.om_package.package_docs import USE_TERMS, VERSION, render_changelog, render_readme
from src.om_package import p10_p11, walk_tables
from src.om_package.walks import load_walks
from src.om_package.wind_regimes import season_regimes, tag_walks, load_campaign, load_climatology
from src.om_package.provenance import read_om_decisions, read_wind_source_manifest
from src.om_package.report_pdf import render_markdown_pdf, render_readme_pdf, report_css
from src.om_package.quality import write_quality_report
from src.om_package.routes import compute_route_geometry_flag, densify_route, route_length_m
from src.om_package.spec import internal_dir_for, render_conformance_markdown, write_conformance
from src.sites.territory import load_territory
from src.om_package.shade import (
    OM2_SHADE_MAX_DIST_M,
    SHADE_STEP_MIN,
    compute_shade_local,
    nodata_floor_m as compute_nodata_floor_m,
)
from src.om_package.sun_envelope import ENVELOPE_SLOT_MIN, route_centroid_latlon
from src.om_package.ventilation import compute_ventilation_proxies
from src.om_package.wind_obs import WINDOW_END, WINDOW_START, cache_paths as wind_cache_paths, fetch_sbgl

from build_om_package_page import build_page as build_om_package_page

ALL_ROUTES = ["OM_1", "OM_2", "OM_3", "OM_4"]
CRS = "EPSG:31983"

#: Label of the geometry epoch of the default inputs (README + manifest). Not
#: a data value: it names what ``Paths`` points at, and is overridden with
#: --geometry-epoch when --buildings/--dtm point at another epoch.
DEFAULT_GEOMETRY_EPOCH = "2019 (cadastral buildings clip, buildings_extended_300m + dtm_extended_300m)"


class EpochPaths(Paths):
    """Paths with the horizon-march geometry inputs overridable: the one
    place a later epoch (2024 ALS DTM + footprints) is swapped in."""

    def __init__(self, root, buildings: str | None = None, dtm: str | None = None):
        super().__init__(root)
        self._buildings = Path(buildings) if buildings else None
        self._dtm = Path(dtm) if dtm else None

    @property
    def buildings_extended_300m(self) -> Path:
        return self._buildings or super().buildings_extended_300m

    @property
    def dtm_extended_300m(self) -> Path:
        return self._dtm or super().dtm_extended_300m


def route_output_dir(om: str, out_dir: Path, internal_dir: Path) -> Path:
    """OM2 goes to the shared package path; every other route goes to the
    internal build directory (PI ruling 2026-09-24: the shared package
    contains OM2 only)."""
    return out_dir if om == "OM_2" else internal_dir


def internal_routes_status(root: Path, version: str) -> str:
    """What actually exists under outputs/_packages/_internal/mare_routes/
    <version>/ right now — never a claim about what the code path CAN do
    (that's always true) versus what it HAS done for this version (audit
    fix, 2026-09-27: the README used to assert OM1/OM3/OM4 land there
    unconditionally, which is false for any version where --route ALL was
    never run)."""
    internal_dir = root / "outputs" / "_packages" / "_internal" / "mare_routes" / version
    if not internal_dir.is_dir():
        return f"not built in this version — no `{internal_dir.relative_to(root).as_posix()}` directory exists yet."
    present = sorted(p.name for p in internal_dir.iterdir() if p.is_dir())
    if not present:
        return f"not built in this version — `{internal_dir.relative_to(root).as_posix()}` exists but is empty."
    return f"on disk in this version under `{internal_dir.relative_to(root).as_posix()}`: {', '.join(present)}."


def route_fetch_date_label(paths: Paths) -> str:
    """Date label for the README's route-JSON 'Date / vintage' cell.
    Prefers a routes manifest.json's own recorded fetch date if one
    exists (same pattern as data/maré/octopus/csv/manifest.json); falls
    back to the route JSON files' own mtimes (audit fix, 2026-09-27: this
    used to be a hand-typed constant, ROUTE_FETCH_DATE, that silently
    drifted every release since nothing re-checked it)."""
    manifest_p = paths.routes_dir / "manifest.json"
    if manifest_p.exists():
        try:
            data = json.loads(manifest_p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        fetched = data.get("fetched_utc") or data.get("fetched") or data.get("fetch_date")
        if fetched:
            return f"fetched {str(fetched)[:10]} (data/maré/octopus/routes/manifest.json)"
    route_files = sorted(paths.routes_dir.glob("OM_*_inferred_route.json"))
    if not route_files:
        return "file dates unavailable — no route JSON files found at build time"
    dates = sorted({datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).date() for p in route_files})
    if len(dates) == 1:
        return f"file dates {dates[0].isoformat()} (route JSON mtimes; no fetch manifest recorded)"
    return f"file dates {dates[0].isoformat()} to {dates[-1].isoformat()} (route JSON mtimes; no fetch manifest recorded)"


def dtm_native_resolution_m(paths: Paths) -> float:
    """The extended DTM's own native pixel size, read from the raster
    (never assumed) — WP-02's build_surface resamples it to 1 m
    (CELL_M) before the horizon march."""
    import rasterio

    with rasterio.open(paths.dtm_extended_300m) as src:
        return abs(src.transform.a)


#: Same pattern the PI runs by hand before a package leaves — kept here so
#: every build re-checks it and writes the hit list next to the package,
#: rather than relying on someone remembering to grep. Hits are NEVER
#: auto-removed: the PI decides each one (p00_disclosure_hits.txt).
DISCLOSURE_PATTERN = re.compile(
    r"party.?wall|dissolve|lancet|nature cities|morphofavela|airflow|brisaverse|"
    r"drive.?sync|solstice|grimmond|\boke\b|sondotecnica|\bIPP\b|mingze|gobatti|fabio",
    re.IGNORECASE,
)


def write_disclosure_hits(out_dir: Path, internal_dir: Path) -> Path:
    """Runs DISCLOSURE_PATTERN over every line of every reader-facing document
    of the built package (shipped README, data dictionary, report, shipped
    scripts) and of the internal CHANGELOG and conformance table, and writes
    (file, line, term, sentence) per hit to p00_disclosure_hits.txt in the
    INTERNAL directory. Hits are reported, never stripped; disclosure is the
    PI's call, not this script's."""
    targets = [
        (out_dir, "README.md"), (internal_dir, "CHANGELOG.md"), (out_dir, "p08_data_dictionary.csv"),
        (out_dir, "report.md"), (internal_dir, "p00_spec_conformance.csv"),
        (out_dir, "OM2/aggregate_to_segments.py"), (out_dir, "OM2/join_shade_example.py"),
    ]
    hits: list[str] = []
    for base, name in targets:
        p = base / name
        if not p.exists():
            continue
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            for m in DISCLOSURE_PATTERN.finditer(line):
                hits.append(f"{name}:{lineno}: [{m.group(0)}] {line.strip()}")
    out_path = internal_dir / "p00_disclosure_hits.txt"
    header = (
        "# Disclosure greplist hits — PI decides each one before this package leaves.\n"
        "# Pattern: party-wall|dissolve|lancet|nature cities|morphofavela|airflow|\n"
        "#          brisaverse|drive-sync|solstice|grimmond|oke|sondotecnica|IPP|\n"
        "#          mingze|gobatti|fabio (case-insensitive; oke and IPP matched as whole words)\n"
        f"# {len(hits)} hit(s) across {', '.join(n for _, n in targets)}.\n\n"
    )
    out_path.write_text(header + ("\n".join(hits) + "\n" if hits else "(no hits)\n"))
    return out_path


def build_one_route(om: str, paths: Paths, out_dir: Path, radii=BUFFER_RADII_M, extras_fn=None) -> dict:
    """Build one route's P-02/P-03/P-04/P-06/P-07 outputs under out_dir.
    out_dir is either the shared package root (for OM2) or the internal
    build directory (for OM1/OM3/OM4) — output_files are reported relative
    to whichever out_dir was passed. ``extras_fn(points_gdf)`` (OM2 only)
    returns (extra_columns keyed by point_id, p07 extras): it runs on the
    fully joined table so its columns are written and quality-checked with
    every other variable."""
    route_json = paths.route_json(om)
    points = densify_route(route_json)
    points["route_geometry_flag"] = compute_route_geometry_flag(points, paths).to_numpy()
    length_m = route_length_m(route_json)

    form = compute_form_variables(points, paths)
    vent = compute_ventilation_proxies(points, form["street_orientation_deg"].to_numpy(), paths)
    nbhd = join_communities(points, paths)

    joined = points.merge(form, on="point_id").merge(vent, on="point_id").merge(nbhd, on="point_id")

    buf = compute_buffer_variables(points, paths, radii=radii)
    joined_with_buf = joined.merge(buf, on="point_id")
    quality_extra = None
    if extras_fn is not None:
        extra_cols, quality_extra = extras_fn(joined_with_buf)
        joined_with_buf = joined_with_buf.merge(extra_cols, on="point_id", how="left")

    route_dir = out_dir / om.replace("OM_", "OM")
    written = write_table(joined_with_buf, route_dir, "points", geo=True)

    variable_cols = [c for c in joined_with_buf.columns if c not in ("point_id", "route_id", "seq", "distance_along_m", "height_m", "geometry")]
    quality = write_quality_report(joined_with_buf, variable_cols, route_dir, extra=quality_extra)

    communities = communities_crossed(points, paths)

    return {
        "route_id": om,
        "length_m": length_m,
        "n_points": len(points),
        "communities_crossed": communities,
        "output_files": [str(p.relative_to(out_dir)) for p in written],
        "quality_summary": {
            "n_points": quality["n_points"],
            "n_columns_checked": len(variable_cols),
            "route_geometry_flagged_points": quality.get("route_geometry_flagged_points"),
        },
    }


LOCAL_TZ = "America/Sao_Paulo"
MATCHED_DIR = Path("data") / "maré" / "octopus" / "prerelease_v020" / "matched"


def write_report_stub(out_dir: Path, version: str) -> tuple[Path, Path]:
    """Placeholder report.md/.pdf: src/om_package/report.py still reads the
    v0.2.0 tables (clock agreement, observed wind) and is rewritten by the
    report lane; until then the package ships this stub, not a stale report."""
    md = out_dir / "report.md"
    md.write_text(
        f"# Octopus OM2 report {version}\n\nPLACEHOLDER: the report for this version is being rewritten. "
        "See README.md for the data description.\n",
        encoding="utf-8",
    )
    pdf = render_markdown_pdf(md, out_dir / "report.pdf", css=report_css(version), title="Octopus OM2 report", md_format="markdown")
    return md, pdf


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="OM2", help="OM1|OM2|OM3|OM4|ALL — non-OM2 routes always go to the internal build dir")
    ap.add_argument("--out", default=None, help="shared package dir for OM2 (default: <root>/outputs/_packages/mare_om2/<version>)")
    ap.add_argument(
        "--internal-out",
        default=None,
        help="internal build dir for OM1/OM3/OM4 (default: <root>/outputs/_packages/_internal/mare_routes/<version>)",
    )
    ap.add_argument("--root", default=str(Paths().root), help="MorphoFavela repo root (absolute)")
    ap.add_argument("--version", default=VERSION)
    ap.add_argument(
        "--matched-dir",
        default=None,
        help="dir of the OM_2_*.csv matched walk files (default: <root>/data/maré/octopus/prerelease_v020/matched)",
    )
    ap.add_argument("--buildings", default=None, help="buildings layer for the horizon march (default: <root>/data/maré/buildings_extended_300m.gpkg)")
    ap.add_argument("--dtm", default=None, help="DTM for the horizon march (default: <root>/data/maré/dtm_extended_300m.tif)")
    ap.add_argument("--geometry-epoch", default=DEFAULT_GEOMETRY_EPOCH, help="label of the geometry epoch, for README + manifest")
    ap.add_argument("--window-start", default=WINDOW_START, help="P-10 season window start (default: the SBGL cache window)")
    ap.add_argument("--window-end", default=WINDOW_END, help="P-10 season window end")
    ap.add_argument("--dose-slot-min", type=int, default=p10_p11.DEFAULT_DOSE_SLOT_MIN, help="slot grid of p10_sun_dose, minutes")
    ap.add_argument("--device", default="cuda", help="torch device for the horizon march")
    ap.add_argument("--skip-page", action="store_true", help="do not rebuild the shared package page (index.html)")
    args = ap.parse_args()

    paths = EpochPaths(args.root, buildings=args.buildings, dtm=args.dtm)
    out_dir = Path(args.out) if args.out else paths.package_dir(args.version)
    internal_dir = (
        Path(args.internal_out)
        if args.internal_out
        else paths.root / "outputs" / "_packages" / "_internal" / "mare_routes" / args.version
    )
    pkg_internal_dir = internal_dir_for(out_dir)

    route_sel = args.route.upper().replace("OM_", "OM")
    routes = ALL_ROUTES if route_sel == "ALL" else [f"OM_{route_sel[2:]}"]

    if "OM_2" in routes:
        out_dir.mkdir(parents=True, exist_ok=True)
        pkg_internal_dir.mkdir(parents=True, exist_ok=True)
    if any(r != "OM_2" for r in routes):
        internal_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "package_version": args.version,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "crs": CRS,
        "use_terms": USE_TERMS,
        "release_scope": "OM2 only. OM1/OM3/OM4 are built by the same code path into an internal directory outside this package.",
        "routes": [],
    }

    ctx: dict = {}
    if "OM_2" in routes:
        matched_dir = Path(args.matched_dir) if args.matched_dir else paths.root / MATCHED_DIR
        print(f"[build_om_package] walks: reading {matched_dir} ...")
        walks_df, fixes = load_walks(matched_dir, paths.route_json("OM_2"))
        walk_dates = sorted(str(d) for d in walks_df["date"].unique())
        if not wind_cache_paths(paths.root)[0].exists():
            print("[build_om_package] SBGL cache missing — fetching from the Iowa ASOS archive ...")
            fetch_sbgl(paths.root, args.window_start, args.window_end)
        season = season_regimes(paths.root)
        regimes = p10_p11.campaign_regime_list(season)
        print(f"[build_om_package] {len(walks_df)} walks on {len(walk_dates)} dates; campaign regimes: "
              + ", ".join(f"{g['name']} {g['mean_direction_deg']:.1f} deg" for g in regimes))
        ctx.update(walks=walks_df, fixes=fixes, walk_dates=walk_dates, season=season, regimes=regimes)

    def om2_extras(points_gdf):
        """P-10/P-11 inputs and the new point columns, from ONE horizon
        march that P-05 shade and p12 reuse below."""
        print(f"[build_om_package] P-10/P-11: horizon march on {args.device} ({len(points_gdf)} points) ...")
        horizon_deg, horizon_az, horizon_tab = p10_p11.horizon_arrays_and_table(points_gdf, paths, device=args.device)
        lat, lon = route_centroid_latlon(points_gdf)
        new_cols = p10_p11.new_point_columns(
            points_gdf, horizon_deg, horizon_az, horizon_tab, regimes=ctx["regimes"], lat=lat, lon=lon
        )
        sun = p10_p11.sun_tables(
            horizon_tab, ctx["walk_dates"], lat=lat, lon=lon, window_start=args.window_start, window_end=args.window_end,
            dose_slot_min=args.dose_slot_min,
        )
        p10_summary = {**sun["summary"], "envelope_slot_min": ENVELOPE_SLOT_MIN}
        ctx.update(
            horizon_deg=horizon_deg, azimuths_deg=horizon_az, horizon_tab=horizon_tab, sun=sun,
            p10_summary=p10_summary, lat=lat, lon=lon, point_ids=points_gdf["point_id"].to_numpy(),
        )
        regime_info = [{k: g[k] for k in ("key", "name", "slug", "mean_direction_deg")} for g in ctx["regimes"]]
        quality_extra = {"p10_p11": {
            "geometry_epoch": args.geometry_epoch,
            "p10": {k: p10_summary[k] for k in ("window", "tz", "n_days", "n_daylight_point_slots", "date_dependent_share",
                                              "class_share_of_daylight", "dose_slot_min")},
            "p11": {"campaign_regimes": regime_info,
                    "note": "ventilation columns are geometry-derived PROXIES; SBGL wind is an airport reference, not wind at the route"},
            "walks": {"n_walks": int(len(ctx["walks"])), "n_walk_dates": len(ctx["walk_dates"]),
                      "n_partial": int(ctx["walks"]["partial"].sum())},
        }}
        return new_cols, quality_extra

    om2_df = None
    for om in routes:
        route_out_dir = route_output_dir(om, out_dir, internal_dir)
        print(f"[build_om_package] {om} -> {route_out_dir} ...")
        result = build_one_route(om, paths, route_out_dir, extras_fn=om2_extras if om == "OM_2" else None)
        if om == "OM_2":
            manifest["routes"].append(result)
            om2_df = pd.read_parquet(route_out_dir / "OM2" / "points.parquet")
        else:
            print(f"  (internal-only, not part of the shared package)")
        print(f"  length_m={result['length_m']:.1f} n_points={result['n_points']} communities={result['communities_crossed']}")

    if om2_df is None:
        print("[build_om_package] OM2 not requested — nothing written to the shared package this run")
        return 0
    assert list(om2_df["point_id"]) == list(ctx["point_ids"]), "points table order differs from the horizon march order"

    # om2_gdf is needed for the nodata floor (README Known limits, manifest
    # p05_shade): a property of the OM2 points against the extended DTM.
    om2_gdf = gpd.GeoDataFrame(
        om2_df[["point_id"]], geometry=gpd.points_from_xy(om2_df["x"], om2_df["y"]), crs=CRS
    )
    print("[build_om_package] P-05: measuring the nodata floor (shade.nodata_floor_m) ...")
    nodata_floor = compute_nodata_floor_m(om2_gdf, paths)
    print(
        f"[build_om_package] P-05: nodata floor min={nodata_floor['min']:.1f}m "
        f"median={nodata_floor['median']:.1f}m max={nodata_floor['max']:.1f}m "
        f"(OM2_SHADE_MAX_DIST_M={OM2_SHADE_MAX_DIST_M:g}m)"
    )
    assert OM2_SHADE_MAX_DIST_M <= nodata_floor["min"], (
        f"OM2_SHADE_MAX_DIST_M={OM2_SHADE_MAX_DIST_M} exceeds the measured nodata floor "
        f"minimum {nodata_floor['min']:.1f}m — the horizon march would hit nodata; revisit "
        "shade.py's OM2_SHADE_MAX_DIST_M before shipping"
    )

    walks_df, walk_dates, regimes, season = ctx["walks"], ctx["walk_dates"], ctx["regimes"], ctx["season"]
    lat, lon = ctx["lat"], ctx["lon"]
    horizon_deg, horizon_az = ctx["horizon_deg"], ctx["azimuths_deg"]

    # P-05: shade on the walk dates, daylight only, Rio local time; parquet only.
    print(f"[build_om_package] P-05: shade on {len(walk_dates)} walk dates ...")
    shade_summary, shade_fig = compute_shade_local(
        om2_df["point_id"], walk_dates, SHADE_STEP_MIN, lat, lon, LOCAL_TZ, horizon_deg, horizon_az,
        out_dir / "p05_building_shade.parquet",
    )
    n_campaign_dates = shade_summary["n_dates"]
    n_shade_rows = shade_summary["n_rows"]
    shade_fraction_daylight_pct = shade_summary["shade_fraction_daylight_pct"]
    print(f"[build_om_package] P-05: {n_shade_rows} rows across {n_campaign_dates} walk dates "
          f"({shade_fraction_daylight_pct}% in building shade, daylight only, {LOCAL_TZ})")
    shade_fig = shade_fig.rename(columns={"timestamp_local": "timestamp"})
    calendar_windows = walks_df.assign(date=walks_df["date"].astype(str)).groupby("date").agg(
        first_timestamp=("start_local", "min"), last_timestamp=("end_local", "max")).reset_index()

    # P-10 / P-11 package-root tables (computed in om2_extras above).
    sun = ctx["sun"]
    write_table(sun["envelope"], out_dir, "p10_sun_envelope")
    sun["dose"].to_parquet(out_dir / "p10_sun_dose.parquet", index=False)  # parquet only: the CSV was 234 MB
    (out_dir / "p10_sun_dose.csv").unlink(missing_ok=True)
    ctx["horizon_tab"].to_parquet(out_dir / "p10_horizon_profiles.parquet", index=False)
    regimes_tbl = p10_p11.wind_regimes_table(season)
    regimes_tbl.to_csv(out_dir / "p11_wind_regimes.csv", index=False)
    by_hour_tbl = p10_p11.regime_by_hour_table(season, paths.root)
    by_hour_tbl.to_csv(out_dir / "p11_regime_by_hour.csv", index=False)

    # P-12: walks and per-walk point values.
    tags = tag_walks(walks_df, load_campaign(paths.root), season["campaign"])
    walks_tbl = walk_tables.walks_table(walks_df, tags)
    write_table(walks_tbl, out_dir, "p02b_walks")
    regime_measures = [f"{stem}_{g['slug']}" for g in regimes for stem in p10_p11.REGIME_MEASURE_STEMS]
    t12 = time.time()
    p12 = walk_tables.walk_points_table(
        om2_df, ctx["fixes"], walks_df, ctx["horizon_tab"], horizon_deg, horizon_az,
        regime_measures=regime_measures, lat=lat, lon=lon,
    )
    write_table(p12, out_dir, "p12_walk_points")
    print(f"[build_om_package] P-12: {len(walks_tbl)} walks, {len(p12)} walk-point rows ({time.time() - t12:.0f} s); "
          f"walks tagged: {walks_tbl['wind_regime'].value_counts().to_dict()}")
    print(
        f"[build_om_package] P-10/P-11: envelope {len(sun['envelope'])} rows, dose {len(sun['dose'])} rows, "
        f"horizon {len(ctx['horizon_tab'])} rows, regimes {len(regimes_tbl)} rows, by-hour {len(by_hour_tbl)} rows"
    )

    # P-08: data dictionary (package-wide, not per-route). OM2/shared only.
    dict_df = dictionary_dataframe(regimes=regimes)
    write_table(dict_df, out_dir, "p08_data_dictionary")

    # Figures. The ventilation figures and the report figure list are the
    # report lane's to redo for two regimes.
    (out_dir / "OM2" / "contact_sheet.png").unlink(missing_ok=True)
    for stale in ("map_vent_shelter.png", "profiles_vent.png", "wind_rose_compare.png"):
        (out_dir / "OM2" / stale).unlink(missing_ok=True)
    try:
        buildings = gpd.read_file(paths.buildings_mare)
    except Exception as exc:  # pragma: no cover - missing source is a build-config error, not a figure bug
        print(f"[build_om_package] WARNING: could not load buildings_mare ({exc}); figures will ship without the building base layer")
        buildings = None
    try:
        territory = load_territory("maré", root=paths.root)
        subunits = territory.subunits
    except Exception as exc:  # pragma: no cover - same: a missing/broken territory registry entry, not a figure bug
        print(f"[build_om_package] WARNING: could not load Maré territory ({exc}); figures will ship without community outlines")
        subunits = None

    fig_dir = out_dir / "OM2"
    for stale in ("map_form.png", "map_shade.png", "profiles.png", "shade_calendar.png", "sun_envelope.png", "sun_dose.png"):
        (fig_dir / stale).unlink(missing_ok=True)
    del shade_fig
    shade_full = load_shade_frame(out_dir / "p05_building_shade.parquet")
    route_total_m = float(om2_df["distance_along_m"].max())
    facts: dict = {"route_length_m": route_total_m, "n_points": int(len(om2_df))}
    build_fig_route(om2_df, buildings, fig_dir / "fig_route.png")
    build_fig_form(om2_df, fig_dir / "fig_form.png")
    build_fig_shade_map(om2_df, shade_full, buildings, fig_dir / "fig_shade_map.png")
    _, facts["shade_calendar"] = build_fig_shade_calendar(shade_full, fig_dir / "fig_shade_calendar.png")
    _, facts["sun_dose"] = build_fig_sun_dose(walks_tbl, p12, route_total_m, fig_dir / "fig_sun_dose.png")
    _, facts["wind"] = build_fig_wind(season, load_campaign(paths.root), load_climatology(paths.root), by_hour_tbl,
                                      fig_dir / "fig_wind.png")
    build_fig_vent_profiles(om2_df, regimes, fig_dir / "fig_vent_profiles.png")
    _, facts["shelter_maps"] = build_fig_shelter_maps(om2_df, regimes, buildings, fig_dir / "fig_shelter_maps.png")
    _, facts["svf_sensor"] = build_fig_svf_sensor(om2_df, walks_tbl, p12, fig_dir / "fig_svf_sensor.png")
    (fig_dir / "figure_facts.json").write_text(json.dumps(facts, indent=2, default=float))
    print(f"[build_om_package] figures written to {fig_dir} (representative walk for the sensor figure: {facts['svf_sensor']['walk_id']})")
    del shade_full

    # The aggregation script and the shade join example travel INSIDE the
    # package, so a recipient with only this directory can re-aggregate (also
    # p12_walk_points per walk, --by walk_id) and exercise the join example.
    shipped_dir = Path(__file__).resolve().parents[1] / "src" / "om_package" / "shipped"
    for shipped_name in ("aggregate_to_segments.py", "join_shade_example.py"):
        shutil.copyfile(shipped_dir / shipped_name, out_dir / "OM2" / shipped_name)
    print(f"[build_om_package] shipped P-03/P-05 scripts into {out_dir / 'OM2'}")

    n_om2_points = len(om2_df)
    n_route_geometry_flagged = int(om2_df["route_geometry_flag"].sum())
    lambda_p_ones = om2_df[om2_df["plan_density_lambda_p"] >= 1.0 - 1e-9]
    n_lambda_p_ones = len(lambda_p_ones)
    n_lambda_p_ones_flagged = int(lambda_p_ones["route_geometry_flag"].sum()) if n_lambda_p_ones else 0
    lambda_p_share_explained_pct = (
        round(100 * n_lambda_p_ones_flagged / n_lambda_p_ones, 1) if n_lambda_p_ones else 0.0
    )
    # The lambda_p==1.0 points NOT explained by route_geometry_flag used to
    # be asserted "plausible fully-built 10 m cells" with no check (audit
    # fix, 2026-09-27) — actually check building_count_buffer_10m > 0 and a
    # recorded building_height_mean_buffer_10m for each of them.
    lambda_p_remainder = lambda_p_ones[~lambda_p_ones["route_geometry_flag"].astype(bool)]
    n_lambda_p_remainder = len(lambda_p_remainder)
    n_lambda_p_remainder_plausible = (
        int(
            (
                (lambda_p_remainder["building_count_buffer_10m"] > 0)
                & lambda_p_remainder["building_height_mean_buffer_10m"].notna()
            ).sum()
        )
        if n_lambda_p_remainder
        else 0
    )

    decisions = read_om_decisions()
    wind_source = read_wind_source_manifest(paths.root)
    internal_status = internal_routes_status(paths.root, args.version)
    fetch_date_label = route_fetch_date_label(paths)
    dtm_res_m = dtm_native_resolution_m(paths)

    readme_kwargs = dict(
        n_om2_points=n_om2_points,
        n_route_geometry_flagged=n_route_geometry_flagged,
        n_lambda_p_ones=n_lambda_p_ones,
        n_lambda_p_ones_flagged=n_lambda_p_ones_flagged,
        lambda_p_share_explained_pct=lambda_p_share_explained_pct,
        n_lambda_p_remainder=n_lambda_p_remainder,
        n_lambda_p_remainder_plausible=n_lambda_p_remainder_plausible,
        route_fetch_date_label=fetch_date_label,
        nodata_floor_m=nodata_floor,
        internal_routes_status=internal_status,
        decisions=decisions,
        dtm_native_resolution_m=dtm_res_m,
        n_walks=len(walks_df),
        n_campaign_dates=n_campaign_dates,
        n_shade_rows=n_shade_rows,
        shade_fraction_daylight_pct=shade_fraction_daylight_pct,
        shade_max_dist_m=OM2_SHADE_MAX_DIST_M,
        p10_summary=ctx["p10_summary"],
        wind_source=wind_source,
        geometry_label=args.geometry_epoch,
        route_length_m=manifest["routes"][0]["length_m"],
        version=args.version,
    )
    changelog_kwargs = dict(version=args.version)
    # Manifest content that does not depend on file hashes goes down BEFORE
    # conformance: P-11's observed-wind part reads provenance.wind_source from
    # it. manifest.json is EXCLUDED from its own file list — audit fix,
    # 2026-09-27: on a rebuild of the same version, manifest.json already
    # exists from the PREVIOUS build, so hashing it would record stale content
    # as its own sha256, a self-hash that could never verify (see
    # io_utils.hash_tree's ``exclude`` docstring). The final write, with the
    # file hashes, is the last thing the build does to the package.
    manifest["p05_shade"] = {
        "n_walks": len(walks_df),
        "n_campaign_dates": n_campaign_dates,
        "n_rows": n_shade_rows,
        "shade_fraction_daylight_pct": shade_fraction_daylight_pct,
        "tz": LOCAL_TZ,
        "max_dist_m": OM2_SHADE_MAX_DIST_M,
        "nodata_floor_m": nodata_floor,
        "campaign_dates": walk_dates,
    }
    manifest["provenance"] = {"decisions": decisions, "wind_source": wind_source}
    manifest["geometry_epoch"] = args.geometry_epoch
    manifest["p10"] = {
        **{k: ctx["p10_summary"][k] for k in ("window", "tz", "n_days", "n_daylight_point_slots", "date_dependent_share",
                                              "class_share_of_daylight", "dose_slot_min", "dose_hours", "envelope_slot_min")},
        "campaign_dates": walk_dates,
    }
    manifest["p11"] = {
        "station": wind_source["station"], "window_utc": wind_source["window_utc"],
        "campaign_regimes": [{k: g[k] for k in ("key", "name", "slug", "mean_direction_deg")} for g in regimes],
        "regimes": regimes_tbl.to_dict(orient="records"),
    }
    manifest["walks"] = {
        "source_dir": MATCHED_DIR.as_posix(), "n_walks": int(len(walks_tbl)), "n_dates": n_campaign_dates,
        "n_partial": int(walks_tbl["partial"].sum()), "n_walk_point_rows": int(len(p12)),
        "tagged_by_regime": walks_tbl["wind_regime"].value_counts().to_dict(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # First pass: README/CHANGELOG without the conformance section, so
    # p00_spec_conformance can be computed over a package directory that
    # already has every other P-01..P-09 artefact (including a README with
    # its required headings) on disk.
    (out_dir / "README.md").write_text(render_readme(**readme_kwargs))
    (pkg_internal_dir / "CHANGELOG.md").write_text(render_changelog(n_om2_points=n_om2_points, **changelog_kwargs))

    # P-00: mechanical conformance to the PI's package spec (P-01..P-11),
    # computed from the files just written — never typed by hand (see
    # src/om_package/spec.py).
    conf = write_conformance(out_dir, out_dir=pkg_internal_dir)
    counts = {s: sum(1 for it in conf["items"] if it["status"] == s)
              for s in ("delivered", "delivered (scoped)", "partial", "pending", "descoped")}
    print(
        "[build_om_package] P-00 spec conformance: "
        + ", ".join(f"{n} {s}" for s, n in counts.items())
        + f" (of {len(conf['items'])})"
    )

    # Second pass: README with the conformance section filled in.
    (out_dir / "README.md").write_text(
        render_readme(conformance_section=render_conformance_markdown(conf) + "\n", **readme_kwargs)
    )

    # README.pdf is rendered from the FINAL README.md, before the hash pass
    # so it ships in the manifest like any other file.
    render_readme_pdf(out_dir)
    print(f"[build_om_package] wrote {out_dir / 'README.pdf'}")

    # manifest: sha256 per file, computed last (over everything just
    # written). manifest.json is EXCLUDED from its own file list — audit
    # fix, 2026-09-27: on a rebuild of the same version, manifest.json
    # already exists on disk from the PREVIOUS build (this run has not
    # written its own copy yet), so hashing out_dir here would capture
    # that stale prior content as manifest.json's own sha256 entry, a
    # self-hash that could never verify. See io_utils.hash_tree's
    # ``exclude`` docstring.
    # report.md/.pdf land before the hash pass so they ship in the manifest.
    try:
        from src.om_package.report import write_report

        write_report(out_dir)
        print(f"[build_om_package] wrote {out_dir / 'report.md'} and {out_dir / 'report.pdf'}")
    except Exception as exc:  # the report is prose on top of the finished data; its failure must not stop the build
        print(f"[build_om_package] WARNING: report not written ({type(exc).__name__}: {exc}); writing the placeholder")
        write_report_stub(out_dir, args.version)
    # Disclosure greplist (PI decides each hit — never auto-removed). Written
    # before hashing: written after, the manifest carried the previous
    # build's hash of this file.
    hits_path = write_disclosure_hits(out_dir, pkg_internal_dir)
    print(f"[build_om_package] wrote disclosure hits to {hits_path}")
    manifest["files"] = hash_tree(out_dir, exclude={"manifest.json"})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build_om_package] wrote manifest to {out_dir / 'manifest.json'}")
    print(
        f"[build_om_package] route_geometry_flag: {n_route_geometry_flagged}/{n_om2_points} OM2 points flagged; "
        f"lambda_p=1.0 explained by flag: {n_lambda_p_ones_flagged}/{n_lambda_p_ones} ({lambda_p_share_explained_pct}%)"
    )

    if args.skip_page:
        print("[build_om_package] --skip-page: shared package page (index.html) not rebuilt")
    else:
        page_path = build_om_package_page(paths.root)
        print(f"[build_om_package] rebuilt package page: {page_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
