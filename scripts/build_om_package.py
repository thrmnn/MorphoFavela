#!/usr/bin/env python3
"""Build the "Maré morphology, OM2" data package — Octopus LRP #2
("Street by street: explaining air temperature differences across streets
and over time in Complexo da Maré", lead Jingxue, PI Simone). Théo is part
of the Octopus team. The package gives street form, sun and wind measures;
the temperature analysis is led by Jingxue. The temperature pairing first
look (src/om_package/temp_pairing.py, fig_temp.py) is kept out of the build
since v1.0.0 (PI 2026-10-06: held for a later version).

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
  P-11 two SBGL wind regimes (wind_regimes, wind_regime_by_hour) and the
       ventilation point columns at each regime (all PROXIES from building
       geometry)
  P-12 walks (walks) and per-walk point values (walk_points)
  P-07 quality report (counts route_geometry_flag)
  P-08 data dictionary — OM2/shared package only
  contact sheet PNG (OM2 only)
  neighbourhoods crossed (printed + saved)
  manifest.json: package_version, crs, use_terms, relative paths, sha256
       per file (computed last, over every file this run wrote)
  outputs/_packages/mare_om2/index.html: the package page (stable URL
       across versions), rebuilt from this run's outputs by
       scripts/build_om_package_page.py — see that module for its content.

Stages (OM2): three compute stages (route + horizon march, shade, walks;
src/om_package/stage_*.py) fill a content-addressed cache under
outputs/_packages/_cache/om2/<stage>/<key>/ (src/om_package/stage_cache.py)
in layout-independent names; the package stage lays the cached tables out
under their layout.py names and writes figures, documents, manifest and ZIP.
A change to layout, docs, figures, manifest or ZIP reruns only the package
stage. --stage package never runs a compute stage and fails on a missing or
stale cache entry; --no-cache recomputes every compute stage.

Run:
    python scripts/build_om_package.py --route OM2 --root /home/theo/SCL/SCR/MorphoFavela
    python scripts/build_om_package.py --route ALL --root /home/theo/SCL/SCR/MorphoFavela
    python scripts/build_om_package.py --stage package --device cpu   # docs-only rebuild, no GPU

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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for build_om_package_page,
# a sibling module — needed when this file is loaded via importlib (tests)
# rather than run as `python scripts/build_om_package.py`.

import geopandas as gpd
import pandas as pd

from src.om_package.buffers import BUFFER_RADII_M
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
from src.om_package import fig_style as fs_style
from src.om_package.fig_flags import build_fig_flags
from src.om_package.vent_schematic import build_fig_vent_schematic
from src.om_package.vent_figures import build_fig_shelter_maps, build_fig_vent_profiles, build_fig_wind
from src.om_package import layout
from src.om_package.io_utils import Paths, hash_tree, write_package_table, write_table
from src.om_package.package_docs import DATA_CREDIT, USE_TERMS, VERSION, frozen_release_error, render_changelog, render_readme
from src.om_package import p10_p11
from src.om_package.wind_regimes import load_campaign, load_climatology
from src.om_package.provenance import read_om_decisions, read_wind_source_manifest
from src.om_package.report_pdf import render_readme_pdf
from src.om_package.quality import write_quality_report
from src.om_package.spec import internal_dir_for, write_conformance
from src.om_package.shade import OM2_SHADE_MAX_DIST_M
from src.om_package.stage_cache import HASH_MEMO, Entry, HashMemo, StageSpec, load_or_run, module_closure, place
from src.om_package.stage_route import compute_route, quality_summary
from src.om_package.stage_shade import LOCAL_TZ
from src.om_package.wind_obs import WINDOW_END, WINDOW_START, cache_paths as wind_cache_paths, fetch_sbgl

from build_om_package_page import build_page as build_om_package_page, write_package_zip

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
    exists; falls
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
        (out_dir, "README.md"), (internal_dir, "CHANGELOG.md"), (out_dir, layout.table("data_dictionary", "csv")),
        (out_dir, "report.md"), (internal_dir, "p00_spec_conformance.csv"),
        (out_dir, layout.SCRIPTS["aggregate_to_segments"]), (out_dir, layout.SCRIPTS["join_shade_example"]),
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


def build_one_route(om: str, paths: Paths, out_dir: Path, radii=BUFFER_RADII_M) -> dict:
    """Build an internal route's (OM1/OM3/OM4) point table and quality
    report under out_dir/OMn. These routes bypass the stage cache; OM2 goes
    through the cached compute stages (stage_route)."""
    table, variable_cols, quality_extra, info = compute_route(om, paths, radii=radii)
    route_dir = out_dir / om.replace("OM_", "OM")
    written = write_table(table, route_dir, layout.INTERNAL_POINTS_STEM, geo=True)
    quality = write_quality_report(table, variable_cols, route_dir, extra=quality_extra)
    info.pop("repair_facts")
    return {**info, "output_files": [str(p.relative_to(out_dir)) for p in written],
            "quality_summary": quality_summary(quality, variable_cols)}


MATCHED_DIR = Path("data") / "maré" / "octopus" / "prerelease_v020" / "matched"
CODE_ROOT = Path(__file__).resolve().parents[1]
STAGE_MODULES = {s: CODE_ROOT / "src" / "om_package" / f"stage_{s}.py" for s in ("route", "shade", "walks")}
#: Left out of the compute stages' code hash: stages write logical names and
#: only the package stage reads the layout, so a layout rename must not rerun
#: the horizon march.
CODE_HASH_EXCLUDE = {CODE_ROOT / "src" / "om_package" / "layout.py"}

#: shipped data files per compute stage: layout key -> extensions, cached as <key>.<ext>
SHIPPED_TABLES: dict[str, dict[str, tuple[str, ...]]] = {
    "route": {"route_points": ("gpkg", "parquet", "csv"), "quality_report": ("json", "csv"),
              "sun_envelope": ("parquet", "csv"), "sun_dose": ("parquet",), "horizon_profiles": ("parquet",)},
    "shade": {"building_shade": ("parquet",)},
    "walks": {"walks": ("parquet", "csv"), "walk_points": ("parquet", "csv"), "wind_regimes": ("csv",),
              "wind_regime_by_hour": ("csv",)},
}


def cache_root_for(root: Path) -> Path:
    return Path(root) / "outputs" / "_packages" / "_cache" / "om2"


def _shapefile(label: str, shp: Path) -> dict[str, Path]:
    return {f"{label}{p.suffix}": p for p in sorted(shp.parent.glob(f"{shp.stem}.*"))}


def stage_specs(paths: Paths, matched_dir: Path, params: dict, upstream_route: str | None = None) -> dict[str, StageSpec]:
    """The OM2 compute stages: inputs, params and code (the import closure of
    each stage module). The torch device is not a param: it decides where
    the march runs, not what it computes."""
    def spec(name, inputs, prm, upstream):
        code = module_closure([STAGE_MODULES[name]], CODE_ROOT, exclude=CODE_HASH_EXCLUDE)
        return StageSpec(name, inputs, prm, code, upstream, CODE_ROOT)

    wind = {"sbgl_campaign": wind_cache_paths(paths.root)[0],
            "sbgl_climatology": paths.root / "data" / "asos" / "SBGL_2015_2024.csv"}
    route_inputs = {
        "route_json": paths.route_json("OM_2"), "matched": matched_dir,
        **_shapefile("buildings_mare", paths.buildings_mare), **_shapefile("street_mare", paths.street_mare),
        "buildings_extended_300m": paths.buildings_extended_300m, "dtm_extended_300m": paths.dtm_extended_300m,
        "features_grid": paths.features_grid, "svf_streets": paths.svf_streets, "hw_streets": paths.hw_streets,
        "neighbourhoods": paths.neighbourhoods_gpkg, **wind,
    }
    up = {"route": upstream_route} if upstream_route else {}
    return {
        "route": spec("route", route_inputs, params, {}),
        "shade": spec("shade", {"dtm_extended_300m": paths.dtm_extended_300m}, {}, up),
        "walks": spec("walks", wind, {}, up),
    }


def resolve_om2_stages(paths: Paths, matched_dir: Path, params: dict, cache_root: Path, *, run: bool,
                       use_cache: bool = True, device: str = "cuda", timings: dict | None = None) -> dict[str, Entry]:
    """Cache entries of the three OM2 compute stages. With run=False no stage
    function is imported or called: a missing or stale entry raises
    StaleCacheError."""
    timings = timings if timings is not None else {}
    memo = HashMemo(cache_root / HASH_MEMO)
    entries: dict[str, Entry] = {}

    def resolve(name, spec, make_fn):
        with timed(f"{'compute' if run else 'cache check'}: {name}", timings):
            entries[name] = load_or_run(spec, cache_root, make_fn() if run else None, use_cache=use_cache, memo=memo)

    def route_fn():
        from src.om_package.stage_route import route_stage
        return lambda work: route_stage(work, paths=paths, matched_dir=matched_dir, device=device, **params)

    resolve("route", stage_specs(paths, matched_dir, params)["route"], route_fn)
    route = entries["route"]
    specs = stage_specs(paths, matched_dir, params, upstream_route=route.key)

    def shade_fn():
        from src.om_package.stage_shade import shade_stage

        def fn(work):
            lat, lon = route.obj("latlon")
            return shade_stage(work, paths=paths, route_points=pd.read_parquet(route.file("route_points.parquet")),
                               walk_dates=route.obj("walk_dates"), lat=lat, lon=lon,
                               horizon_deg=route.obj("horizon_deg"), azimuths_deg=route.obj("azimuths_deg"))
        return fn

    def walks_fn():
        from src.om_package.stage_walks import walks_stage

        def fn(work):
            lat, lon = route.obj("latlon")
            return walks_stage(
                work, root=paths.root, route_points=pd.read_parquet(route.file("route_points.parquet")),
                walks=route.obj("walks"), fixes=route.obj("walk_fixes"), season=route.obj("season"),
                regimes=route.obj("regimes"), horizon_tab=route.obj("horizon_tab"),
                horizon_deg=route.obj("horizon_deg"), azimuths_deg=route.obj("azimuths_deg"), lat=lat, lon=lon)
        return fn

    resolve("shade", specs["shade"], shade_fn)
    resolve("walks", specs["walks"], walks_fn)
    return entries


def lay_out_tables(entries: dict[str, Entry], out_dir: Path) -> None:
    """Shipped data files from the cache to their layout.py names (hard
    links when the cache and the package share a filesystem)."""
    for stage, tables in SHIPPED_TABLES.items():
        for key, exts in tables.items():
            for ext in exts:
                place(entries[stage].file(f"{key}.{ext}"), layout.table_path(out_dir, key, ext))


@contextmanager
def timed(label: str, timings: dict):
    t0 = time.time()
    try:
        yield
    finally:
        timings[label] = time.time() - t0
        print(f"[build_om_package] time {label}: {timings[label]:.1f} s")


def main() -> int:
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
    ap.add_argument("--dose-slot-min", type=int, default=p10_p11.DEFAULT_DOSE_SLOT_MIN, help="slot grid of sun_dose, minutes")
    ap.add_argument("--device", default="cuda", help="torch device for the horizon march")
    ap.add_argument("--skip-page", action="store_true", help="do not rebuild the shared package page (index.html)")
    ap.add_argument(
        "--stage", choices=("all", "compute", "package"), default="all",
        help="compute: fill the OM2 stage cache only; package: lay out the package from the cache only "
             "(never runs a compute stage, fails on a missing or stale entry); all: both",
    )
    ap.add_argument("--no-cache", action="store_true", help="recompute every OM2 compute stage and replace its cache entry")
    args = ap.parse_args()
    if args.stage == "package" and args.no_cache:
        ap.error("--no-cache recomputes; it cannot be combined with --stage package")

    t_build = time.time()
    timings: dict[str, float] = {}
    paths = EpochPaths(args.root, buildings=args.buildings, dtm=args.dtm)
    out_dir = Path(args.out) if args.out else paths.package_dir(args.version)
    internal_dir = (
        Path(args.internal_out)
        if args.internal_out
        else paths.root / "outputs" / "_packages" / "_internal" / "mare_routes" / args.version
    )
    pkg_internal_dir = internal_dir_for(out_dir)
    cache_root = cache_root_for(paths.root)
    frozen = frozen_release_error(out_dir, paths.package_dir(args.version).parent)
    if frozen:
        print(f"[build_om_package] REFUSED: {frozen} (target {out_dir})", file=sys.stderr)
        return 2

    route_sel = args.route.upper().replace("OM_", "OM")
    routes = ALL_ROUTES if route_sel == "ALL" else [f"OM_{route_sel[2:]}"]

    internal = [r for r in routes if r != "OM_2"]
    if internal and args.stage == "package":
        print(f"[build_om_package] --stage package: internal routes {internal} are not cached; not rebuilt")
    elif internal:
        internal_dir.mkdir(parents=True, exist_ok=True)
        for om in internal:
            print(f"[build_om_package] {om} -> {internal_dir} (internal-only, not part of the shared package) ...")
            with timed(f"internal route {om}", timings):
                result = build_one_route(om, paths, internal_dir)
            print(f"  length_m={result['length_m']:.1f} n_points={result['n_points']} communities={result['communities_crossed']}")

    if "OM_2" not in routes:
        print("[build_om_package] OM2 not requested — nothing written to the shared package this run")
        return 0

    matched_dir = Path(args.matched_dir) if args.matched_dir else paths.root / MATCHED_DIR
    params = {"geometry_epoch": args.geometry_epoch, "window_start": args.window_start,
              "window_end": args.window_end, "dose_slot_min": args.dose_slot_min}
    run = args.stage != "package"
    if run and not wind_cache_paths(paths.root)[0].exists():
        print("[build_om_package] SBGL cache missing — fetching from the Iowa ASOS archive ...")
        fetch_sbgl(paths.root, args.window_start, args.window_end)
    entries = resolve_om2_stages(paths, matched_dir, params, cache_root, run=run, use_cache=not args.no_cache,
                                 device=args.device, timings=timings)
    if args.stage == "compute":
        print(f"[build_om_package] --stage compute: cache filled under {cache_root}; package not laid out")
        _print_timings(timings, t_build)
        return 0

    with timed("package: data layout", timings):
        # A rebuild of a version must not leave files of an earlier layout behind.
        for sub in (layout.DATA_DIR, layout.FIGURES_DIR, layout.SCRIPTS_DIR, "OM2"):
            shutil.rmtree(out_dir / sub, ignore_errors=True)
        for legacy in out_dir.glob("p[0-9][0-9]*"):
            legacy.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        pkg_internal_dir.mkdir(parents=True, exist_ok=True)
        lay_out_tables(entries, out_dir)

    route, shade, walks = entries["route"], entries["shade"], entries["walks"]
    route_result = route.obj("route_result")
    repair_facts = route.obj("repair_facts")
    regimes, season, walk_dates = route.obj("regimes"), route.obj("season"), route.obj("walk_dates")
    shade_summary, nodata_floor = shade.obj("shade_summary"), shade.obj("nodata_floor")
    walks_summary = walks.obj("walks_summary")
    n_walks = len(route.obj("walks"))
    om2_df = pd.read_parquet(layout.table_path(out_dir, "route_points", "parquet"))
    walks_tbl = pd.read_parquet(layout.table_path(out_dir, "walks", "parquet"))
    p12 = pd.read_parquet(layout.table_path(out_dir, "walk_points", "parquet"))
    by_hour_tbl = walks.obj("wind_regime_by_hour")
    print(f"  OM_2 length_m={route_result['length_m']:.1f} n_points={route_result['n_points']} "
          f"communities={route_result['communities_crossed']}")

    manifest = {
        "package_version": args.version,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "crs": CRS,
        "data_credit": DATA_CREDIT,
        "use_terms": USE_TERMS,
        "release_scope": "OM2 only. OM1/OM3/OM4 are built by the same code path into an internal directory outside this package.",
        "routes": [{
            **{k: route_result[k] for k in ("route_id", "length_m", "n_points", "communities_crossed")},
            "output_files": [layout.table("route_points", ext) for ext in route_result["route_point_exts"]],
            "quality_summary": route_result["quality_summary"],
        }],
    }

    # P-08: data dictionary (package-wide, not per-route). OM2/shared only.
    write_package_table(dictionary_dataframe(regimes=regimes), out_dir, "data_dictionary")

    # Figures. The ventilation figures and the report figure list are the
    # report lane's to redo for two regimes.
    t_fig = time.time()
    try:
        buildings = gpd.read_file(paths.buildings_mare)
    except Exception as exc:  # pragma: no cover - missing source is a build-config error, not a figure bug
        print(f"[build_om_package] WARNING: could not load buildings_mare ({exc}); figures will ship without the building base layer")
        buildings = None

    fig_dir = out_dir / layout.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    shade_full = load_shade_frame(layout.table_path(out_dir, "building_shade", "parquet"))
    route_total_m = float(om2_df["distance_along_m"].max())
    facts: dict = {"route_length_m": route_total_m, "n_points": int(len(om2_df)),
                   **fs_style.flag_facts(fs_style.flagged_spans(om2_df))}
    build_fig_route(om2_df, buildings, fig_dir / layout.FIG["route"])
    build_fig_form(om2_df, fig_dir / layout.FIG["form"])
    build_fig_shade_map(om2_df, shade_full, buildings, fig_dir / layout.FIG["shade_map"])
    _, facts["shade_calendar"] = build_fig_shade_calendar(shade_full, fig_dir / layout.FIG["shade_calendar"])
    _, facts["sun_dose"] = build_fig_sun_dose(walks_tbl, p12, route_total_m, fig_dir / layout.FIG["sun_dose"])
    _, facts["wind"] = build_fig_wind(season, load_campaign(paths.root), load_climatology(paths.root), by_hour_tbl,
                                      fig_dir / layout.FIG["wind"])
    build_fig_vent_profiles(om2_df, regimes, fig_dir / layout.FIG["vent_profiles"])
    _, facts["shelter_maps"] = build_fig_shelter_maps(om2_df, regimes, buildings, fig_dir / layout.FIG["shelter_maps"])
    _, facts["svf_sensor"] = build_fig_svf_sensor(om2_df, walks_tbl, p12, fig_dir / layout.FIG["svf_sensor"])
    build_fig_vent_schematic(fig_dir / layout.FIG["vent_schematic"],
                             route_median_ratio=float(om2_df["height_width_ratio"].median()))
    build_fig_flags(route.obj("repair_res"), buildings, route.obj("repair_fixes"), fig_dir / layout.FIG["flags"])
    facts["flags"] = repair_facts
    (out_dir / layout.FIGURE_FACTS).write_text(json.dumps(facts, indent=2, default=float))
    print(f"[build_om_package] figures written to {fig_dir} (representative walk for the sensor figure: {facts['svf_sensor']['walk_id']})")
    del shade_full
    timings["package: figures"] = time.time() - t_fig
    print(f"[build_om_package] time package: figures: {timings['package: figures']:.1f} s")

    # The aggregation script and the shade join example travel INSIDE the
    # package, so a recipient with only this directory can re-aggregate (also
    # walk_points per walk, --by walk_id) and exercise the join example.
    shipped_dir = Path(__file__).resolve().parents[1] / "src" / "om_package" / "shipped"
    for rel in layout.SCRIPTS.values():
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shipped_dir / Path(rel).name, out_dir / rel)
    print(f"[build_om_package] shipped scripts into {out_dir / layout.SCRIPTS_DIR}")

    n_om2_points = len(om2_df)
    n_route_geometry_flagged = int(om2_df["route_geometry_flag"].sum())
    lambda_p_ones = om2_df[om2_df["plan_density_lambda_p"] >= 1.0 - 1e-9]
    n_lambda_p_ones = len(lambda_p_ones)
    n_lambda_p_ones_flagged = int(lambda_p_ones["route_geometry_flag"].sum()) if n_lambda_p_ones else 0
    lambda_p_share_explained_pct = (
        round(100 * n_lambda_p_ones_flagged / n_lambda_p_ones, 1) if n_lambda_p_ones else 0.0
    )
    decisions = read_om_decisions()
    wind_source = read_wind_source_manifest(paths.root)

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
        "n_walks": n_walks,
        "n_campaign_dates": shade_summary["n_dates"],
        "n_rows": shade_summary["n_rows"],
        "shade_fraction_daylight_pct": shade_summary["shade_fraction_daylight_pct"],
        "tz": LOCAL_TZ,
        "max_dist_m": OM2_SHADE_MAX_DIST_M,
        "nodata_floor_m": nodata_floor,
        "campaign_dates": walk_dates,
    }
    manifest["provenance"] = {"decisions": decisions, "wind_source": wind_source}
    manifest["geometry_epoch"] = args.geometry_epoch
    p10_summary = route.obj("p10_summary")
    manifest["p10"] = {
        **{k: p10_summary[k] for k in ("window", "tz", "n_days", "n_daylight_point_slots", "date_dependent_share",
                                       "class_share_of_daylight", "dose_slot_min", "dose_hours", "envelope_slot_min")},
        "campaign_dates": walk_dates,
    }
    manifest["p11"] = {
        "station": wind_source["station"], "window_utc": wind_source["window_utc"],
        "campaign_regimes": [{k: g[k] for k in ("key", "name", "slug", "mean_direction_deg")} for g in regimes],
        "regimes": walks.obj("wind_regimes_records"),
    }
    manifest["walks"] = {
        "source_dir": MATCHED_DIR.as_posix(), "n_walks": walks_summary["n_walks"], "n_dates": shade_summary["n_dates"],
        "n_partial": walks_summary["n_partial"], "n_walk_point_rows": walks_summary["n_walk_point_rows"],
        "tagged_by_regime": walks_summary["tagged_by_regime"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    t_docs = time.time()
    # First pass: README/CHANGELOG without the conformance section, so
    # p00_spec_conformance can be computed over a package directory that
    # already has every other P-01..P-09 artefact (including a README with
    # its required headings) on disk.
    (out_dir / "README.md").write_text(render_readme(out_dir))
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
    from src.om_package.report import write_report

    write_report(out_dir)
    print(f"[build_om_package] wrote {out_dir / 'report.md'} and {out_dir / 'report.pdf'}")
    # Disclosure greplist (PI decides each hit — never auto-removed). Written
    # before hashing: written after, the manifest carried the previous
    # build's hash of this file.
    hits_path = write_disclosure_hits(out_dir, pkg_internal_dir)
    print(f"[build_om_package] wrote disclosure hits to {hits_path}")
    timings["package: documents"] = time.time() - t_docs
    print(f"[build_om_package] time package: documents: {timings['package: documents']:.1f} s")
    with timed("package: manifest", timings):
        manifest["files"] = hash_tree(out_dir, exclude={"manifest.json"})
        # Only manifest.json carries the cache keys: written after every
        # document, so no other shipped file changes when a key does.
        manifest["cache"] = {name: {"key": e.key, "git_head": e.meta.get("git_head"), "dirty": e.meta.get("dirty")}
                             for name, e in entries.items()}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build_om_package] wrote manifest to {out_dir / 'manifest.json'}")
    print(
        f"[build_om_package] route_geometry_flag: {n_route_geometry_flagged}/{n_om2_points} OM2 points flagged; "
        f"lambda_p=1.0 explained by flag: {n_lambda_p_ones_flagged}/{n_lambda_p_ones} ({lambda_p_share_explained_pct}%)"
    )

    with timed("package: zip", timings):
        zip_path = write_package_zip(out_dir)
    print(f"[build_om_package] wrote {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")

    if args.skip_page:
        print("[build_om_package] --skip-page: shared package page (index.html) not rebuilt")
    else:
        with timed("package: page", timings):
            page_path = build_om_package_page(paths.root)
        print(f"[build_om_package] rebuilt package page: {page_path}")

    _print_timings(timings, t_build)
    return 0


def _print_timings(timings: dict, t_build: float) -> None:
    print("[build_om_package] timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items())
          + f"; total {time.time() - t_build:.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
