#!/usr/bin/env python3
"""Build the canonical street-observer network per site.

Reads the existing ``svf_streets.gpkg`` under
``outputs/{site}/morphometrics/svf/``, drops pipeline-internal columns,
adds a stable ``point_id``, and writes the cleaned observer network to a
new first-class location:

    outputs/{site}/sampling_streets/
    ├── observers.gpkg       — for QGIS / shapefile-style consumers (EPSG:31983)
    ├── observers.parquet    — fast Python joins (recommended for big sites, EPSG:31983)
    ├── observers.geojson    — RFC-7946 compliant WGS84 lon/lat, for web tooling
    └── manifest.json        — sampling parameters, CRS, source provenance

This is Step 1 of the observer-network reorg: it produces the new layer
without touching any code that currently reads ``svf_streets.gpkg``.
Downstream rewiring (SVF and solar reading from the new location) is a
separate follow-up.

Usage::

    python scripts/build_street_observers.py                   # all 5 campaign sites
    python scripts/build_street_observers.py --site vidigal
    python scripts/build_street_observers.py --bundle          # also produce the tarball
    python scripts/build_street_observers.py --bundle-only     # skip rebuild, just bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import geopandas as gpd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.svf_v2.paths import resolve_boundary, resolve_paths  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("build_street_observers")

CAMPAIGN_SITES = ("vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré")

# Sampler defaults baked into the existing svf_streets.gpkg files.
# These match the argparse defaults in scripts/run_svf_v2.py at the time
# of sampling (commit history confirms no overrides were used for the
# campaign sites).
SAMPLER_PARAMS = {
    "centerline_spacing_m": 1.5,
    "pedestrian_height_m": 1.5,
    "building_safety_margin_m": 0.5,
    "boundary_clipped": True,
}

# Bundle root for collaborator hand-off.  Versioned so future
# regenerations don't silently overwrite a sent-out drop.
BUNDLE_VERSION = "v1"
BUNDLE_ROOT = PROJECT_ROOT / "outputs" / "_distribution" / f"street_observers_{BUNDLE_VERSION}"

# Map internal site keys to ASCII folder names for the collaborator bundle.
SITE_BUNDLE_NAME = {"maré": "mare"}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _git_head_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _today_iso() -> str:
    # Date only — avoids a fresh ISO timestamp on every rerun, which
    # would churn manifest diffs even when nothing semantic changed.
    from datetime import date

    return date.today().isoformat()


def build_observers(site: str) -> Path:
    """Read svf_streets.gpkg for ``site`` and write the cleaned observer files.

    Returns the path to the site's ``sampling_streets/`` directory.
    """
    in_path = (
        PROJECT_ROOT / "outputs" / site / "morphometrics" / "svf" / "svf_streets.gpkg"
    )
    if not in_path.exists():
        raise FileNotFoundError(f"Source svf_streets.gpkg not found: {in_path}")

    out_dir = PROJECT_ROOT / "outputs" / site / "sampling_streets"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] reading %s", site, in_path)
    src_gdf = gpd.read_file(in_path)
    n = len(src_gdf)
    if n == 0:
        raise ValueError(f"{site}: source file has 0 rows")

    # Stable, site-prefixed, zero-padded id.  Width = max(6, digits(N)).
    width = max(6, len(str(n)))
    point_ids = [f"{site}_{i:0{width}d}" for i in range(n)]

    xs = np.array([p.x for p in src_gdf.geometry], dtype=np.float64)
    ys = np.array([p.y for p in src_gdf.geometry], dtype=np.float64)

    out_gdf = gpd.GeoDataFrame(
        {
            "point_id": point_ids,
            "street_id": src_gdf["street_id"].astype("int64").to_numpy(),
            "distance_along_m": src_gdf["distance_along"].astype("float64").to_numpy(),
            "x_m": xs,
            "y_m": ys,
            "z_terrain_m": src_gdf["z"].astype("float64").to_numpy(),
            "z_observer_m": src_gdf["z_observer"].astype("float64").to_numpy(),
            "was_offset": src_gdf["was_offset"].astype(bool).to_numpy(),
            "offset_distance_m": src_gdf["offset_distance"].astype("float64").to_numpy(),
            "geometry": src_gdf.geometry.values,
        },
        crs=src_gdf.crs,
    )

    gpkg_path = out_dir / "observers.gpkg"
    parquet_path = out_dir / "observers.parquet"
    geojson_path = out_dir / "observers.geojson"
    if gpkg_path.exists():
        gpkg_path.unlink()
    out_gdf.to_file(gpkg_path, driver="GPKG", layer="observers")
    out_gdf.to_parquet(parquet_path, index=False)

    # GeoJSON: RFC 7946 mandates WGS84 lon/lat.  We reproject for the
    # export but keep the x_m / y_m columns at their original UTM values
    # so a downstream consumer can recover the projected coordinates
    # without a second reprojection.
    if geojson_path.exists():
        geojson_path.unlink()
    out_gdf.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")

    # Manifest
    try:
        dtm_path, _fp_path, roads_path = resolve_paths(site)
    except FileNotFoundError as exc:
        logger.warning("[%s] could not resolve source paths: %s", site, exc)
        dtm_path = roads_path = None
    boundary_path = resolve_boundary(site)

    n_offset = int(out_gdf["was_offset"].sum())
    manifest = {
        "site": site,
        "n_observers": n,
        "crs": out_gdf.crs.to_string() if out_gdf.crs else None,
        "schema": {
            "point_id": "str — stable site-prefixed zero-padded id",
            "street_id": "int — id of the source road feature this point was sampled from",
            "distance_along_m": "float — metres from the start of the source LineString",
            "x_m": "float — easting (CRS units, same as geometry)",
            "y_m": "float — northing (CRS units, same as geometry)",
            "z_terrain_m": "float — DTM elevation at (x, y)",
            "z_observer_m": "float — z_terrain_m + pedestrian_height_m",
            "was_offset": "bool — point was pushed out of a building footprint at sampling time",
            "offset_distance_m": "float — distance from original sample location (0 when !was_offset)",
            "geometry": "Point — same (x, y) as columns, kept for GIS tooling",
        },
        "sampler": SAMPLER_PARAMS,
        "qa": {
            "n_offset": n_offset,
            "offset_fraction": round(n_offset / n, 4),
        },
        "source": {
            "svf_streets_gpkg": str(in_path.relative_to(PROJECT_ROOT)),
            "roads_file": str(roads_path.relative_to(PROJECT_ROOT)) if roads_path else None,
            "dtm_file": str(dtm_path.relative_to(PROJECT_ROOT)) if dtm_path else None,
            "boundary_file": (
                str(boundary_path.relative_to(PROJECT_ROOT)) if boundary_path else None
            ),
        },
        "provenance": {
            "generated": _today_iso(),
            "sampler_code_commit": _git_head_short(),
            "builder_script": "scripts/build_street_observers.py",
        },
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    logger.info(
        "[%s] wrote %d observers  (offset: %d / %.1f%%)  -> %s",
        site,
        n,
        n_offset,
        100.0 * n_offset / n,
        out_dir.relative_to(PROJECT_ROOT),
    )
    return out_dir


def _write_bundle_readme(path: Path, sites_built: list[str]) -> None:
    lines = [
        f"# MorphoFavela — street observers, {BUNDLE_VERSION}",
        "",
        "Per-site street-level observer networks sampled along road",
        "centerlines.  Each observer is a pedestrian-eye location (1.5 m",
        "above terrain) used downstream to evaluate sky-view factor and",
        "seasonal solar access.  This bundle contains the *sampling*",
        "geometry only; SVF and solar values are distributed separately.",
        "",
        "## Sites",
        "",
    ]
    for site in sites_built:
        bundle_name = SITE_BUNDLE_NAME.get(site, site)
        lines.append(f"- `{bundle_name}/` — internal key `{site}`")
    lines += [
        "",
        "## Files per site",
        "",
        "- `observers.gpkg` — GeoPackage, layer `observers`.  Opens in QGIS,",
        "  ArcGIS, or any GDAL-aware tool.",
        "- `observers.parquet` — same data as a flat table.  Fast load in",
        "  Python (`geopandas.read_parquet` or `pandas.read_parquet`).  The",
        "  geometry column is WKB-encoded per GeoParquet 1.0.",
        "- `observers.geojson` — same observers in **WGS84 lon/lat** per",
        "  RFC 7946.  The `x_m` / `y_m` properties stay in their original",
        "  UTM values so you can recover the projected coords without a",
        "  second reprojection.  Loads in web tools (Mapbox, Leaflet,",
        "  Kepler, geojson.io) without any setup.",
        "- `manifest.json` — sampling parameters, CRS, source-file paths,",
        "  generation date, and code commit.",
        "",
        "## Schema",
        "",
        "| column | type | unit | meaning |",
        "|---|---|---|---|",
        "| `point_id` | str | — | stable site-prefixed id, e.g. `vidigal_000001` |",
        "| `street_id` | int64 | — | id of the source road feature |",
        "| `distance_along_m` | float64 | m | metres from start of source LineString |",
        "| `x_m`, `y_m` | float64 | m | coords in EPSG:31983 (SIRGAS 2000 / UTM 23S) |",
        "| `z_terrain_m` | float64 | m | DTM elevation at the point |",
        "| `z_observer_m` | float64 | m | `z_terrain_m` + 1.5 m pedestrian height |",
        "| `was_offset` | bool | — | sample was pushed out of a building footprint |",
        "| `offset_distance_m` | float64 | m | how far it moved (0 when `!was_offset`) |",
        "| `geometry` | Point | — | same `(x_m, y_m)`, kept for GIS tooling |",
        "",
        "## CRS",
        "",
        "- `observers.gpkg` / `observers.parquet`: **EPSG:31983** (SIRGAS",
        "  2000 / UTM zone 23S).  Units are metres.",
        "- `observers.geojson`: **EPSG:4326** (WGS84 lon/lat) per RFC 7946.",
        "",
        "All three formats describe the same observers — pick by tooling,",
        "not by content.",
        "",
        "## Sampling parameters",
        "",
        f"- Centerline spacing: **{SAMPLER_PARAMS['centerline_spacing_m']} m**",
        f"- Pedestrian height: **{SAMPLER_PARAMS['pedestrian_height_m']} m**",
        f"- Building safety margin: **{SAMPLER_PARAMS['building_safety_margin_m']} m**",
        "- Road centerlines clipped to the site boundary before sampling",
        "  (no observers on highways crossing outside the favela edge).",
        "- Points falling inside a building footprint are pushed to the",
        "  nearest exterior + the safety margin; the move is logged in",
        "  `was_offset` and `offset_distance_m`.",
        "",
        "## QA flag — when to filter on `was_offset`",
        "",
        "Most points (>97%) are not offset.  The flag exists because road",
        "centerlines and building footprints don't always align: in a few",
        "spots a road crosses through a footprint vertex or a footprint",
        "overlaps the carriageway.  The offset preserves the observer count",
        "but moves the sample by a small distance.  If you want only",
        "strictly on-centerline points, filter `was_offset == False`.",
        "",
        "## Contact",
        "",
        "Generated by `scripts/build_street_observers.py` in the",
        "MorphoFavela repo.  Questions: thermann.ai@gmail.com.",
        "",
    ]
    path.write_text("\n".join(lines))


def build_bundle(sites: list[str]) -> Path:
    """Stage the collaborator bundle and write a tar.gz alongside it."""
    if BUNDLE_ROOT.exists():
        shutil.rmtree(BUNDLE_ROOT)
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

    built: list[str] = []
    for site in sites:
        src_dir = PROJECT_ROOT / "outputs" / site / "sampling_streets"
        if not src_dir.exists():
            logger.warning("Skipping %s in bundle: %s missing", site, src_dir)
            continue
        bundle_name = SITE_BUNDLE_NAME.get(site, site)
        dst_dir = BUNDLE_ROOT / bundle_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("observers.gpkg", "observers.parquet", "observers.geojson", "manifest.json"):
            src = src_dir / fname
            if not src.exists():
                logger.warning("  %s/%s missing -- skipping", site, fname)
                continue
            shutil.copy2(src, dst_dir / fname)
        built.append(site)

    _write_bundle_readme(BUNDLE_ROOT / "README.md", built)

    # Tarball.  Sorted file list for deterministic archives.
    tar_path = BUNDLE_ROOT.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    entries = sorted(BUNDLE_ROOT.rglob("*"))
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(BUNDLE_ROOT, arcname=BUNDLE_ROOT.name, recursive=False)
        for entry in entries:
            arc = Path(BUNDLE_ROOT.name) / entry.relative_to(BUNDLE_ROOT)
            tf.add(entry, arcname=str(arc), recursive=False)

    sha = _sha256(tar_path)
    logger.info("Bundle: %s  (%d sites, sha256=%s)", tar_path, len(built), sha[:16])
    return tar_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--site",
        action="append",
        default=None,
        help="Site to build (repeatable). Default: all 5 campaign sites.",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="After rebuilding, stage the collaborator bundle + tarball.",
    )
    parser.add_argument(
        "--bundle-only",
        action="store_true",
        help="Skip rebuild and only re-stage the bundle from existing outputs.",
    )
    args = parser.parse_args()

    sites = args.site if args.site else list(CAMPAIGN_SITES)

    if not args.bundle_only:
        failures: list[tuple[str, str]] = []
        for site in sites:
            try:
                build_observers(site)
            except Exception as exc:  # surfaced to CLI
                logger.exception("[%s] failed: %s", site, exc)
                failures.append((site, repr(exc)))
        if failures:
            logger.warning("%d site(s) failed:", len(failures))
            for site, msg in failures:
                logger.warning("  %s -- %s", site, msg)

    if args.bundle or args.bundle_only:
        build_bundle(sites)

    return 0


if __name__ == "__main__":
    sys.exit(main())
