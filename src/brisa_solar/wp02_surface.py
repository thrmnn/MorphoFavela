"""WP-02: obstruction surface (DTM + rasterised building tops).

Spec: docs/wp02_horizon_engine_spec.md §1. The per-patch visibility engine
(wp02_horizon.py) needs one elevation grid to march rays over; this module
builds it once so the footprint rasterisation is not repeated per observer.

Run: python3 -m src.brisa_solar.wp02_surface DTM FOOTPRINTS CELL_M OUT_STEM
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import MergeAlg, rasterize
from rasterio.warp import reproject

from .constants import REPO_ROOT, load_params


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:8]


def _read_and_resample(dtm_path: Path, cell_m: float):
    """DTM at its native resolution, bilinearly resampled to cell_m if they differ."""
    with rasterio.open(dtm_path) as src:
        native_res = abs(src.transform.a)
        crs = src.crs
        nodata = src.nodata
        if abs(native_res - cell_m) < 1e-9:
            arr = src.read(1).astype("float32")
            transform = src.transform
        else:
            scale = native_res / cell_m
            width = max(1, round(src.width * scale))
            height = max(1, round(src.height * scale))
            transform = rasterio.transform.from_bounds(*src.bounds, width, height)
            arr = np.empty((height, width), dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=crs,
                dst_transform=transform,
                dst_crs=crs,
                src_nodata=nodata,
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )
    if nodata is not None:
        arr = np.where(np.isclose(arr, nodata, rtol=1e-3), np.nan, arr)
    return arr, transform, crs


def build_surface(dtm_path, footprints_path, cell_m: float, out_path, all_touched: bool = False) -> Path:
    """Build `surface = max(dtm, building_top)` and an `is_building` mask.

    `all_touched` (rasterio.features.rasterize) marks every cell a footprint
    polygon *touches*, not just cells whose centre falls inside it — lets
    thin building parts (a wall, an eave) survive rasterization at coarse
    cell sizes, at the cost of slightly over-stating building footprint area.

    Writes `<out_path>_surface.tif`, `<out_path>_is_building.tif`, and
    `<out_path>_meta.json`. Returns the surface GeoTIFF path.
    """
    dtm_path = Path(dtm_path)
    footprints_path = Path(footprints_path)
    out_path = Path(out_path)

    dtm, transform, crs = _read_and_resample(dtm_path, cell_m)

    gdf = gpd.read_file(footprints_path)
    if gdf.crs is not None and crs is not None and str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)

    fp_params = load_params()["footprints"]
    base_attr, alt_attr, top_attr = fp_params["base_attr"], fp_params["height_attr"], fp_params["top_attr"]

    base = gdf[base_attr].to_numpy(dtype="float64")
    altura = gdf[alt_attr].to_numpy(dtype="float64")
    topo = gdf[top_attr].to_numpy(dtype="float64")
    top = np.where(np.isfinite(topo) & (topo > base), topo, base + altura)

    valid = gdf.geometry.notna() & ~gdf.geometry.is_empty
    order = np.argsort(np.where(valid, top, -np.inf))  # ascending; rasterize(replace) keeps the last write
    shapes = [
        (gdf.geometry.iloc[i], float(top[i])) for i in order if valid.iloc[i]
    ]

    if shapes:
        building_top = rasterize(
            shapes,
            out_shape=dtm.shape,
            transform=transform,
            fill=np.nan,
            all_touched=all_touched,
            dtype="float32",
            merge_alg=MergeAlg.replace,
        )
    else:
        building_top = np.full(dtm.shape, np.nan, dtype="float32")

    is_building = np.isfinite(building_top)
    surface = np.where(is_building, np.fmax(dtm, building_top), dtm).astype("float32")

    # building_id: 1-based positional index into `gdf` (0 = no building), rasterized
    # with the SAME order/filter/merge rule as building_top so a cell's owning id
    # always agrees with which building's height won that cell (WP-04F spec §1) —
    # this is what lets patch_visibility recognise "this cell is the observer's own
    # building" rather than a neighbour's, at the 1 m cell size where a façade point
    # inset just outside a wall can still nearest-round onto its own roof cell.
    id_shapes = [(gdf.geometry.iloc[i], int(i) + 1) for i in order if valid.iloc[i]]
    if id_shapes:
        building_id = rasterize(
            id_shapes,
            out_shape=dtm.shape,
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="int32",
            merge_alg=MergeAlg.replace,
        )
    else:
        building_id = np.zeros(dtm.shape, dtype="int32")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    surface_tif = out_path.with_name(out_path.stem + "_surface.tif")
    building_tif = out_path.with_name(out_path.stem + "_is_building.tif")
    building_id_tif = out_path.with_name(out_path.stem + "_building_id.tif")
    ground_tif = out_path.with_name(out_path.stem + "_ground.tif")
    meta_json = out_path.with_name(out_path.stem + "_meta.json")

    profile = dict(
        driver="GTiff", height=surface.shape[0], width=surface.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=np.nan,
    )
    with rasterio.open(surface_tif, "w", **profile) as dst:
        dst.write(surface, 1)
    bprofile = dict(profile, dtype="uint8", nodata=None)
    with rasterio.open(building_tif, "w", **bprofile) as dst:
        dst.write(is_building.astype("uint8"), 1)
    idprofile = dict(profile, dtype="int32", nodata=0)
    with rasterio.open(building_id_tif, "w", **idprofile) as dst:
        dst.write(building_id.astype("int32"), 1)
    with rasterio.open(ground_tif, "w", **profile) as dst:
        dst.write(dtm.astype("float32"), 1)

    bounds = rasterio.transform.array_bounds(surface.shape[0], surface.shape[1], transform)
    meta = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dtm_path": str(dtm_path),
        "footprints_path": str(footprints_path),
        "cell_m": cell_m,
        "all_touched": bool(all_touched),
        "shape": list(surface.shape),
        "bounds": list(bounds),
        "crs": str(crs),
        "n_features": int(len(gdf)),
        "n_features_rasterized": int(len(shapes)),
        "top_rule": f"{top_attr} when finite and > {base_attr}, else {base_attr} + {alt_attr}",
        "git_sha": _git_sha(),
        "md5_dtm": _md5(dtm_path),
        "md5_footprints": _md5(footprints_path),
    }
    meta_json.write_text(json.dumps(meta, indent=1))
    return surface_tif


def load_surface(surface_tif, is_building_tif=None):
    """Read a surface GeoTIFF (and optional is_building mask) back into arrays."""
    with rasterio.open(surface_tif) as src:
        surface = src.read(1)
        transform = src.transform
        crs = src.crs
    is_building = None
    if is_building_tif is not None and Path(is_building_tif).exists():
        with rasterio.open(is_building_tif) as src:
            is_building = src.read(1).astype(bool)
    return surface, transform, crs, is_building


def load_building_id(building_id_tif) -> np.ndarray | None:
    """Read a `_building_id.tif` (WP-04F) back into an int32 array, or None if absent —
    kept as a standalone loader (not folded into load_surface's return tuple) so the
    5 existing 4-tuple call sites of load_surface are untouched."""
    p = Path(building_id_tif)
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        return src.read(1).astype("int32")


def load_ground(ground_tif) -> np.ndarray | None:
    """Read a `_ground.tif` (WP-04F, the bare DTM on the surface grid, no building
    tops) back into a float32 array, or None if absent."""
    p = Path(ground_tif)
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        return src.read(1).astype("float32")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dtm_path")
    ap.add_argument("footprints_path")
    ap.add_argument("cell_m", type=float)
    ap.add_argument("out_stem")
    args = ap.parse_args()
    path = build_surface(args.dtm_path, args.footprints_path, args.cell_m, Path(args.out_stem))
    print(json.dumps({"surface": str(path)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
