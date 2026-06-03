"""Onboard a new favela from the citywide IPP cadaster + DTM.

For each --favela NAME (must match Favelas_Limit_2019 `nome`), this
script materialises the on-disk structure expected by ``resolve_paths``:

    data/{site}/raw/
        {site}_buildings.shp   # bbox-filtered IPP cadaster, restricted to boundary buffer
        {site}_boundary.shp    # boundary polygon from Favelas_Limit_2019
        roads_{site}.shp       # Logradouros clipped to boundary buffer
        {site}_dtm.tif         # DTM_RJ.tif clipped to boundary + 150 m buffer

Then PATCHES ``src/svf_v2/paths.py`` (the AREA_FILES registry +
SUPPORTED_AREAS list in src/config.py) to register the new site.

The site is now ready for:
    python scripts/build_extended_context.py --area {site}
    python scripts/run_morphometric_audit.py --area {site}

The DTM is clipped programmatically (NOT manually in QGIS). For batch
onboarding the trade-off the README warns about is worth it.

Slugification: name → snake_case lowercased without diacritics.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import from_bounds

_ROOT = Path(__file__).resolve().parents[2]
RJ = _ROOT / "data" / "RJ"
BLD_SHP = RJ / "buildings_RJ_2019.shp"
DTM_TIF = RJ / "DTM_RJ.tif"
FAV_SHP = RJ / "Favelas_Limit_2019.shp"
ROADS_SHP = RJ / "Logradouros.shp"

CONFIG_PY = _ROOT / "src" / "config.py"
PATHS_PY = _ROOT / "src" / "svf_v2" / "paths.py"

BUFFER_M = 150.0  # DTM clip buffer (matches existing site DTM convention)


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s


def materialize_site(name: str) -> dict:
    """Build data/{slug}/raw/ from citywide layers for favela `name`."""
    favs = gpd.read_file(FAV_SHP)
    match = favs[favs["nome"] == name]
    if match.empty:
        return {"name": name, "ok": False, "reason": "name not in Favelas_Limit_2019"}
    if len(match) > 1:
        match = match.iloc[[match.geometry.area.idxmax()]]
    boundary = match.copy()
    geom = boundary.geometry.iloc[0]
    slug = slugify(name)

    raw_dir = _ROOT / "data" / slug / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Boundary shapefile
    boundary_out = raw_dir / f"{slug}_boundary.shp"
    boundary[["nome", "geometry"]].to_file(boundary_out)

    # Buildings — bbox-filter the 2.4M-row citywide cadaster then clip.
    minx, miny, maxx, maxy = geom.buffer(BUFFER_M).bounds
    bld = gpd.read_file(BLD_SHP, bbox=(minx, miny, maxx, maxy))
    bld = bld[bld.intersects(geom)].copy()
    invalid = ~bld.geometry.is_valid
    if invalid.any():
        bld.loc[invalid, "geometry"] = bld.loc[invalid, "geometry"].buffer(0)
    # Force 2D — IPP geom is XYZ; downstream tools expect XY.
    bld["geometry"] = bld.geometry.force_2d()
    buildings_out = raw_dir / f"{slug}_buildings.shp"
    bld.to_file(buildings_out)

    # Roads (Logradouros) — clip to boundary + buffer
    try:
        rds = gpd.read_file(ROADS_SHP, bbox=(minx, miny, maxx, maxy))
        rds = rds[rds.intersects(geom.buffer(BUFFER_M))].copy()
        rds["geometry"] = rds.geometry.force_2d()
        roads_out = raw_dir / f"roads_{slug}.shp"
        rds.to_file(roads_out)
        n_roads = int(len(rds))
    except Exception as e:
        n_roads = -1
        roads_out = None

    # DTM — clip with rasterio.mask to the buffered boundary polygon.
    dtm_out = raw_dir / f"{slug}_dtm.tif"
    with rasterio.open(DTM_TIF) as src:
        win = from_bounds(minx, miny, maxx, maxy, src.transform)
        clip_geom = [geom.buffer(BUFFER_M).__geo_interface__]
        out_img, out_transform = rio_mask(src, clip_geom, crop=True, all_touched=True)
        meta = src.meta.copy()
        meta.update(
            height=out_img.shape[1],
            width=out_img.shape[2],
            transform=out_transform,
            compress="lzw",
        )
        with rasterio.open(dtm_out, "w", **meta) as dst:
            dst.write(out_img)

    return {
        "name": name,
        "slug": slug,
        "ok": True,
        "n_buildings": int(len(bld)),
        "n_roads": n_roads,
        "boundary": str(boundary_out.relative_to(_ROOT)),
        "buildings": str(buildings_out.relative_to(_ROOT)),
        "roads": str(roads_out.relative_to(_ROOT)) if roads_out else None,
        "dtm": str(dtm_out.relative_to(_ROOT)),
    }


def register_in_config(slug: str) -> None:
    """Append slug to SUPPORTED_AREAS and INFORMAL_AREAS in src/config.py
    if not already present. Idempotent."""
    text = CONFIG_PY.read_text()
    for list_name in ("SUPPORTED_AREAS", "INFORMAL_AREAS"):
        pat = re.compile(rf"{list_name}\s*=\s*\[(.*?)\]", re.DOTALL)
        m = pat.search(text)
        if not m:
            continue
        body = m.group(1)
        if f'"{slug}"' in body:
            continue
        # Insert just before the closing bracket
        new_body = body.rstrip()
        if not new_body.endswith(","):
            new_body += ","
        new_body += f'\n    "{slug}",\n'
        text = text[: m.start(1)] + new_body + text[m.end(1) :]
    CONFIG_PY.write_text(text)


def register_in_paths(slug: str) -> None:
    """Insert an AREA_FILES entry for the new slug. Idempotent."""
    text = PATHS_PY.read_text()
    if f'"{slug}":' in text:
        return
    entry = (
        f'    "{slug}": {{\n'
        f'        "dtm": "{slug}_dtm.tif",\n'
        f'        "footprints": "{slug}_buildings.shp",\n'
        f'        "roads": "roads_{slug}.shp",\n'
        f'        "boundary": "{slug}_boundary.shp",\n'
        f"    }},\n"
    )
    # Insert just before the closing `}` of AREA_FILES.
    pat = re.compile(r"(AREA_FILES\s*=\s*\{.*?)(\n\}\n)", re.DOTALL)
    m = pat.search(text)
    if not m:
        raise RuntimeError("Could not locate AREA_FILES closing brace in paths.py")
    text = text[: m.end(1)] + "\n" + entry + m.group(2) + text[m.end(2) :]
    PATHS_PY.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--favela", action="append", required=True,
                    help="Exact name from Favelas_Limit_2019.nome; repeatable")
    args = ap.parse_args()

    results = []
    for name in args.favela:
        print(f"\n=== {name} ===")
        try:
            rec = materialize_site(name)
        except Exception as e:
            rec = {"name": name, "ok": False, "reason": f"materialize error: {e}"}
        print(rec)
        if rec.get("ok"):
            register_in_config(rec["slug"])
            register_in_paths(rec["slug"])
            print(f"Registered in config.py and svf_v2/paths.py as '{rec['slug']}'")
        results.append(rec)

    print("\n--- Summary ---")
    for r in results:
        flag = "OK" if r.get("ok") else "FAIL"
        print(f"[{flag}] {r.get('name')} → {r.get('slug', '-')}")


if __name__ == "__main__":
    main()
