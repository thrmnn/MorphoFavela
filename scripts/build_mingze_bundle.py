"""Build the 3D-data bundle for Mingze's solar-access pipeline.

Per-site directory with DTM raster, 3D building footprints (altura column
gives meters above base), site boundary, and canonical observer points
in three formats. ASCII-renames maré → mare so the tarball is
cross-platform safe. Writes a top-level README, manifest.json with
per-file sha256, and an alongside .tar.gz + .sha256.

Run: python scripts/build_mingze_bundle.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITES = ["rocinha", "complexo_do_alemao", "riodaspedras", "maré"]
SITE_BUNDLE_NAME = {"maré": "mare"}
BUNDLE_VERSION = "v1"
OUT_ROOT = REPO / "outputs" / "_distribution"
BUNDLE_DIR = OUT_ROOT / f"mingze_3d_bundle_{BUNDLE_VERSION}"
TARBALL = OUT_ROOT / f"mingze_3d_bundle_{BUNDLE_VERSION}.tar.gz"

SHP_SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qmd")


def site_files(site: str) -> dict[str, Path]:
    raw = REPO / "data" / site / "raw"
    obs = REPO / "outputs" / site / "sampling_streets"
    # AREA_FILES registry equivalents
    reg = {
        "rocinha": {
            "dtm": "rocinha_dtm.tif",
            "buildings": "rocinha_buildings.shp",
            "boundary": "rocinha_boundary.shp",
        },
        "complexo_do_alemao": {
            "dtm": "complexo_do_alemao_dtm.tif",
            "buildings": "complexo_do_alemao_buildings.shp",
            "boundary": "complexo_do_alemao_boundary.shp",
        },
        "riodaspedras": {
            "dtm": "riodaspedras_dtm.tif",
            "buildings": "riodaspedras_buildings.shp",
            "boundary": "riodaspedras_boundary.shp",
        },
        "maré": {
            "dtm": "mare_dtm.tif",
            "buildings": "buildings_mare.shp",
            "boundary": "mare_boundary.shp",
        },
    }[site]
    return {
        "dtm": raw / reg["dtm"],
        "buildings_shp": raw / reg["buildings"],
        "boundary_shp": raw / reg["boundary"],
        "observers_gpkg": obs / "observers.gpkg",
        "observers_geojson": obs / "observers.geojson",
        "observers_parquet": obs / "observers.parquet",
        "observers_manifest": obs / "manifest.json",
    }


def copy_shp_with_sidecars(shp: Path, dest_dir: Path, *, rename: str | None = None) -> list[Path]:
    written = []
    stem = shp.stem
    new_stem = rename or stem
    for ext in SHP_SIDECARS:
        sidecar = shp.with_suffix(ext)
        if sidecar.exists():
            target = dest_dir / f"{new_stem}{ext}"
            shutil.copy2(sidecar, target)
            written.append(target)
    return written


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


SCHEMA_MD_PER_SITE = """# {full_name} — file dictionary

CRS for every spatial layer: **EPSG:31983** (SIRGAS 2000 / UTM 23S), units = metres.

## dtm.tif
- Digital terrain model raster. One band, float32 metres above SIRGAS 2000 ellipsoid.
- Pixel size 5 m.
- Use for ground elevation at observer points (already encoded as `z_terrain_m` in the observer table; this raster is provided for any custom resampling).

## terrain.stl
- Triangulated terrain mesh built from `dtm.tif` at full resolution, binary STL,
  **world coordinates** (EPSG:31983 metres — no local-origin translation).

## buildings_3d.stl
- All buildings extruded on top of the terrain: base elevation from the registry
  `base` field (DTM-sampled where missing), extrusion height = `altura`. Binary STL,
  world coordinates.
- `terrain.stl + buildings_3d.stl` together are **exactly the scene MorphoFavela's
  ray-caster used** for the SVF/solar numbers we compare against — simulating on this
  geometry removes the model-construction difference between our pipelines.

## buildings.shp (+ .dbf .shx .prj .cpg)
- 3D Polygon geometry. Per-feature attributes from the Rio municipal `edificações` registry:
  - `altura` — building height above its base, metres. Use this as extrusion height for 3D modelling.
  - `base` — base elevation of the building footprint, m a.s.l.
  - `topo` — top elevation = base + altura, m a.s.l.
  - `tipo` — registry category (mostly "Edificação" residential / commercial).
- The 3D Polygon already carries Z, so a parser that respects POLYHEDRALSURFACE or 2.5D polygons can extrude directly.
- For Ladybug: extrude the planar footprint by `altura` and offset the base to `base`.

## boundary.shp
- Single-polygon site outline. Use to clip the building set or to bound the Ladybug study area.

## observers.gpkg / observers.geojson / observers.parquet
Same data, three formats. EPSG:31983 for gpkg/parquet, WGS84 for geojson (RFC 7946).
Schema:
- `point_id` — stable, site-prefixed, zero-padded (e.g. `rocinha_038690`).
- `x_m`, `y_m` — projected coordinates (EPSG:31983), metres.
- `z_terrain_m` — terrain elevation under the point, m a.s.l.
- `z_observer_m` — pedestrian-eye height = `z_terrain_m` + 1.5 m. **Use this Z when placing a sun-access sensor.**
- `street_id` — road segment the observer was sampled on (Rio `logradouros` registry index).
- `distance_along_m` — arclength along the road segment, m.
- `was_offset`, `offset_distance_m` — whether the point was nudged off a building footprint and by how much.

## observers_manifest.json
Per-site sampling parameters (spacing, pedestrian height, building safety margin) + counts + git sha.
"""


def build_site_stls(site: str, files: dict[str, Path], site_dir: Path) -> None:
    """terrain.stl + buildings_3d.stl in world coordinates (EPSG:31983).

    Their union is byte-for-byte the scene MorphoFavela's SVF/solar
    ray-caster sees — sending it removes the model-construction confound
    from cross-method comparisons.
    """
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.svf_v2 import build_building_meshes, build_terrain_mesh

    terrain = build_terrain_mesh(files["dtm"])
    terrain_path = site_dir / "terrain.stl"
    terrain.save(str(terrain_path), binary=True)
    print(f"    terrain.stl: {terrain.n_cells:,} cells, {terrain_path.stat().st_size / 1e6:.1f} MB")

    buildings, _ = build_building_meshes(files["buildings_shp"], files["dtm"], area=site)
    if buildings is None:
        raise RuntimeError(f"{site}: no valid buildings for STL extrusion")
    bld_path = site_dir / "buildings_3d.stl"
    buildings.save(str(bld_path), binary=True)
    print(f"    buildings_3d.stl: {buildings.n_cells:,} cells, {bld_path.stat().st_size / 1e6:.1f} MB")


def write_per_site_schema(dest: Path, site: str) -> Path:
    full_name = {
        "rocinha": "Rocinha",
        "complexo_do_alemao": "Complexo do Alemão",
        "riodaspedras": "Rio das Pedras",
        "maré": "Maré",
    }[site]
    md = SCHEMA_MD_PER_SITE.format(full_name=full_name)
    p = dest / "SCHEMA.md"
    p.write_text(md)
    return p


README = """# MorphoFavela — 3D data bundle for Mingze (Ladybug solar-access)

**Bundle version:** {version}
**Built:** {built}
**Git sha:** {git_sha}
**Sites:** {sites_human}
**CRS for every spatial layer:** EPSG:31983 (SIRGAS 2000 / UTM 23S), units = metres.
**Vidigal is intentionally NOT in this bundle** (you have already run it).

## What's here, per site

```
{site_slug}/
  dtm.tif              # terrain raster, 1-band float32 metres, 5 m pixels
  terrain.stl          # ready-made terrain mesh (binary STL, EPSG:31983 world coords)
  buildings_3d.stl     # buildings extruded on the terrain (binary STL, world coords)
  buildings.shp+.dbf+.shx+.prj+.cpg   # 3D polygons, has 'altura' (height, m)
  boundary.shp+...     # site outline
  observers.gpkg       # canonical sampling points, EPSG:31983
  observers.geojson    # same data, WGS84
  observers.parquet    # same data, EPSG:31983
  observers_manifest.json
  SCHEMA.md            # per-file dictionary, attribute units
```

## To build a Ladybug 3D model

**Shortcut (recommended):** import `terrain.stl` + `buildings_3d.stl` directly —
they are the exact occlusion scene our ray-caster used, so any remaining
difference between your results and ours is down to the solver, not the model.
Coordinates are world EPSG:31983 metres (large values — translate to a local
origin inside Rhino if your tolerance settings complain, but apply the same
translation to the observer points).

Building the model yourself instead:

1. Read `buildings.shp` (it's a 3D Polygon registry). For Ladybug:
   - Extrude each footprint by the `altura` attribute (metres).
   - Offset the base by the `base` attribute (m above EPSG:31983 ellipsoid datum) if needed; the planar polygon Z already encodes this on most sites.
2. Read `dtm.tif` to add a terrain mesh (use rasterio or rhino-grasshopper plug-in).
3. Read `observers.gpkg` (or geojson if you prefer WGS84). Place a solar-access sensor at each point at the height `z_observer_m` (= `z_terrain_m + 1.5` m, pedestrian eye level).
4. Run your seasonal sun-path. Output schema we'd love to match for direct comparison:
   - `point_id` (matches our table)
   - `solar_hours_winter`, `solar_hours_summer`, `solar_hours_equinox_mar`, `solar_hours_equinox_sep`, `solar_hours_annual` (mean of the four)
   - `irradiance_*_wh` per same dates if available
   - `sunshine_ratio_mean` (fraction of geometric daylight reached)

If you can write back to CSV with `point_id + solar_hours_*` we can join 1:1 against our results and produce a side-by-side accuracy plot for the paper.

## Caveats

- These shapefiles are the Rio municipal `edificações` v2024 registry. The `altura` attribute is height above the building base, not above sea level (use `topo` for absolute top elevation).
- The observer set is the canonical street-level sampling (1.5 m spacing along road centerlines, pedestrian height = DTM + 1.5 m, building safety margin 0.5 m). See `observers_manifest.json` per site for the exact parameters.
- For Maré, the in-folder name is `mare/` (ASCII) but the site is referred to in our reports as `Maré`.
- For Rocinha and Complexo do Alemão the road shapefile omits stairway features (Escadaria / Ladeira); pedestrian circulation through stairs is therefore under-sampled.

## Provenance

See `manifest.json` next to this README for per-file sha256 and source paths.

## Bundle license

The shapefiles are open-data from the Rio municipality (`logradouros` + `edificações` registry). DTM derived from project-specific airborne LiDAR (per-site provenance, contact us for the upstream source). Observer points are MorphoFavela-derived under the same license as the repo.

Contact: Theo Hermann (thermann.ai@gmail.com).
"""


def main() -> int:
    print(f"Building bundle at {BUNDLE_DIR}")
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "built_iso": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "sites": {},
    }

    for site in SITES:
        slug = SITE_BUNDLE_NAME.get(site, site)
        site_dir = BUNDLE_DIR / slug
        site_dir.mkdir(parents=True)
        files = site_files(site)
        missing = [name for name, p in files.items() if not p.exists()]
        if missing:
            print(f"  [warn] {site}: missing {missing}", file=sys.stderr)

        # DTM
        if files["dtm"].exists():
            dst = site_dir / "dtm.tif"
            shutil.copy2(files["dtm"], dst)

        # Buildings + sidecars (rename to canonical name)
        if files["buildings_shp"].exists():
            copy_shp_with_sidecars(files["buildings_shp"], site_dir, rename="buildings")

        # Boundary + sidecars
        if files["boundary_shp"].exists():
            copy_shp_with_sidecars(files["boundary_shp"], site_dir, rename="boundary")

        # STL scene (terrain + extruded buildings, world coordinates)
        if files["dtm"].exists() and files["buildings_shp"].exists():
            build_site_stls(site, files, site_dir)

        # Observers (3 formats + manifest)
        for src_key, dst_name in [
            ("observers_gpkg", "observers.gpkg"),
            ("observers_geojson", "observers.geojson"),
            ("observers_parquet", "observers.parquet"),
            ("observers_manifest", "observers_manifest.json"),
        ]:
            if files[src_key].exists():
                shutil.copy2(files[src_key], site_dir / dst_name)

        write_per_site_schema(site_dir, site)

        # Per-file sha256
        site_files_listing = {}
        for p in sorted(site_dir.iterdir()):
            if p.is_file():
                site_files_listing[p.name] = {
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_of(p),
                }
        manifest["sites"][slug] = {
            "source_site_key": site,
            "n_files": len(site_files_listing),
            "files": site_files_listing,
        }
        total_mb = sum(f["size_bytes"] for f in site_files_listing.values()) / (1024 * 1024)
        print(f"  {slug}: {len(site_files_listing)} files, {total_mb:.1f} MB")

    # Top-level README + manifest
    sites_human = ", ".join(SITE_BUNDLE_NAME.get(s, s) for s in SITES)
    (BUNDLE_DIR / "README.md").write_text(
        README.format(
            version=BUNDLE_VERSION,
            built=manifest["built_iso"],
            git_sha=manifest["git_sha"],
            sites_human=sites_human,
            site_slug="<site>",
        )
    )
    (BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Tarball
    print(f"Creating {TARBALL}")
    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as tar:
        tar.add(BUNDLE_DIR, arcname=BUNDLE_DIR.name)

    sha = sha256_of(TARBALL)
    (TARBALL.with_suffix(".tar.gz.sha256")).write_text(f"{sha}  {TARBALL.name}\n")
    tar_mb = TARBALL.stat().st_size / (1024 * 1024)
    print(f"Tarball: {TARBALL} ({tar_mb:.1f} MB)")
    print(f"sha256:  {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
