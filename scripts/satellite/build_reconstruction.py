"""Free-EO 2.5D reconstruction of one favela — no ground truth, ever.

LEAKAGE RULE (from docs/satellite_reconstruction_plan.md): this module must
never open anything under ``data/RJ/``. Its only local input is the neutral AOI
written by ``export_aoi.py``; everything else is downloaded from free, no-auth
sources. Scoring against the IPP answer key happens in ``score_vs_ipp.py``.
A test (``tests/test_satellite_leakage.py``) enforces this statically.

Components:
  DTM        Copernicus GLO-30 (anonymous S3 COG).  Note this is a DSM used as
             a DTM proxy — the plan's "borrowed 30 m" — which is exactly why a
             positive elevation bias is expected in dense cores.
  Footprints Google Open Buildings v3 via the VIDA GeoParquet mirror on
             source.coop, filtered to ``bf_source == 'google'`` because that
             mirror is a Google+Microsoft union.
  Heights    Open Buildings 2.5D Temporal (2019).  BLOCKED — no anonymous
             mirror exists (probed 2026-09-08: the public `open-buildings-data`
             bucket holds only v1/v2/v3 polygons; `open-buildings-temporal*`
             buckets 404), so the layer needs Google Earth Engine auth, which
             is the user's to drive.  Per the plan we stop at footprints+DTM
             and leave heights NULL rather than substituting a guess.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds

REPO = Path(__file__).resolve().parents[2]
WORKING_CRS = "EPSG:31983"

COP_DEM_ROOT = "https://copernicus-dem-30m.s3.amazonaws.com"
OB_PARQUET = (
    "https://data.source.coop/vida/google-microsoft-open-buildings/"
    "geoparquet/by_country/country_iso=BRA/BRA.parquet"
)
HEIGHTS_BLOCKED_REASON = (
    "Open Buildings 2.5D Temporal (2019) has no anonymous mirror; the layer "
    "requires Google Earth Engine authentication, which is the user's to drive."
)
# The AOI is small; 300 m of slack keeps edge buildings and DEM cells whole.
AOI_BUFFER_M = 300.0
# A wider DTM window so the scorer can estimate the vertical-reference offset on
# open flat ground outside the favela. Same free product, wider window — no
# ground truth is involved, so the quarantine is untouched.
CONTROL_BUFFER_M = 2000.0


def _cop_tile_url(lat_sw: int, lon_sw: int) -> str:
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    name = f"Copernicus_DSM_COG_10_{ns}{abs(lat_sw):02d}_00_{ew}{abs(lon_sw):03d}_00_DEM"
    return f"{COP_DEM_ROOT}/{name}/{name}.tif"


def _tiles_for_bounds(bounds_4326: tuple[float, float, float, float]) -> list[str]:
    x0, y0, x1, y1 = bounds_4326
    urls = []
    for lat in range(math.floor(y0), math.floor(y1) + 1):
        for lon in range(math.floor(x0), math.floor(x1) + 1):
            urls.append(_cop_tile_url(lat, lon))
    return urls


def paste_native_tiles(windows: list[tuple]) -> tuple[np.ndarray, object, float]:
    """Combine per-tile windows on their shared native lattice, without resampling.

    All GLO-30 tiles sit on the same global 1/3600-degree grid, so overlapping
    windows can be pasted by integer offset. Nothing is interpolated here; the
    single resampling step happens when the scorer projects onto the IPP grid.

    ``rasterio.merge(bounds=...)`` was used here first and is wrong for this
    job: it snaps to its own grid and shifted elevations by up to 27 m against a
    direct source-to-target reprojection (measured 2026-09-08).
    """
    res = abs(windows[0][1].a)
    if any(abs(abs(t.a) - res) > 1e-12 or abs(abs(t.e) - res) > 1e-12 for _, t, _, _ in windows):
        raise SystemExit("tiles are not on a common lattice; cannot paste without resampling")

    extents = [rasterio.transform.array_bounds(a.shape[0], a.shape[1], t) for a, t, _, _ in windows]
    ux0 = min(e[0] for e in extents)
    uy1 = max(e[3] for e in extents)
    uw = int(round((max(e[2] for e in extents) - ux0) / res))
    uh = int(round((uy1 - min(e[1] for e in extents)) / res))

    native = np.full((uh, uw), np.nan, dtype="float32")
    for (arr, _, _, nodata_i), (ex0, _, _, ey1) in zip(windows, extents):
        arr = arr.astype("float32")
        if nodata_i is not None:
            arr = np.where(arr == nodata_i, np.nan, arr)
        r0 = int(round((uy1 - ey1) / res))
        c0 = int(round((ex0 - ux0) / res))
        block = native[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]]
        native[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]] = np.where(
            np.isfinite(arr), arr, block
        )
    return native, rasterio.transform.from_origin(ux0, uy1, res, res), res


def fetch_dtm(
    aoi: gpd.GeoDataFrame,
    out_tif: Path,
    provenance: dict,
    buffer_m: float = AOI_BUFFER_M,
    key: str = "dtm",
) -> Path:
    """Window-read GLO-30 over the AOI and reproject to the working CRS."""
    buf = aoi.buffer(buffer_m)
    bounds_4326 = tuple(gpd.GeoSeries(buf, crs=aoi.crs).to_crs("EPSG:4326").total_bounds)
    urls = _tiles_for_bounds(bounds_4326)

    windows = []
    for url in urls:
        with rasterio.open(f"/vsicurl/{url}") as src:
            window = from_bounds(*bounds_4326, transform=src.transform).round_offsets()
            window = window.round_lengths()
            if window.width <= 0 or window.height <= 0:
                continue
            windows.append(
                (src.read(1, window=window), src.window_transform(window), src.crs, src.nodata)
            )
    if not windows:
        raise SystemExit("no GLO-30 pixels intersect the AOI")
    src_crs = windows[0][2]
    native, native_transform, res = paste_native_tiles(windows)
    uh, uw = native.shape

    native_tif = out_tif.with_name(out_tif.stem + "_native.tif")
    native_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        native_tif,
        "w",
        driver="GTiff",
        height=uh,
        width=uw,
        count=1,
        dtype="float32",
        crs=src_crs,
        transform=native_transform,
        nodata=np.nan,
        compress="deflate",
    ) as nds:
        nds.write(native, 1)

    # Convenience copy in the project CRS; the scorer uses the native grid.
    dst_transform, width, height = calculate_default_transform(
        src_crs, WORKING_CRS, uw, uh, *rasterio.transform.array_bounds(uh, uw, native_transform)
    )
    dst = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=native,
        destination=dst,
        src_transform=native_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=WORKING_CRS,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=WORKING_CRS,
        transform=dst_transform,
        nodata=np.nan,
        compress="deflate",
    ) as dstds:
        dstds.write(dst, 1)

    provenance[key] = {
        "product": "Copernicus GLO-30 DSM (used as DTM proxy)",
        "urls": urls,
        "license": "Copernicus DEM open licence (free, redistributable)",
        "native_crs": str(src_crs),
        "native_grid_file": native_tif.name,
        "native_shape": [int(uh), int(uw)],
        "output_crs": WORKING_CRS,
        "output_shape": [int(height), int(width)],
        "output_res_m": [abs(dst_transform.a), abs(dst_transform.e)],
        "resampling": "bilinear",
        "buffer_m": buffer_m,
    }
    print(
        f"DTM[{key}]: native {uh}x{uw} @ {res:.6f} deg -> {native_tif.name}; "
        f"reprojected {height}x{width} -> {out_tif.name}"
    )
    return native_tif


def _row_group_bboxes(pf) -> np.ndarray:
    """Per-row-group spatial extent from the GeoParquet 1.1 bbox covering column."""
    md = pf.metadata
    paths = {md.schema.column(i).path: i for i in range(md.num_columns)}
    cols = [paths[f"bbox.{k}"] for k in ("xmin", "ymin", "xmax", "ymax")]
    out = np.empty((md.num_row_groups, 4), dtype="float64")
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        out[rg, 0] = g.column(cols[0]).statistics.min
        out[rg, 1] = g.column(cols[1]).statistics.min
        out[rg, 2] = g.column(cols[2]).statistics.max
        out[rg, 3] = g.column(cols[3]).statistics.max
    return out


def fetch_footprints(aoi: gpd.GeoDataFrame, out_gpkg: Path, provenance: dict) -> gpd.GeoDataFrame:
    """Range-read only the Open Buildings row groups whose bbox meets the AOI."""
    import fsspec
    import pyarrow.parquet as pq

    buf = gpd.GeoSeries(aoi.buffer(AOI_BUFFER_M), crs=aoi.crs)
    qx0, qy0, qx1, qy1 = buf.to_crs("EPSG:4326").total_bounds

    t0 = time.time()
    handle = fsspec.open(OB_PARQUET, "rb", block_size=8 * 1024 * 1024).open()
    pf = pq.ParquetFile(handle)
    bx = _row_group_bboxes(pf)
    hit = np.where((bx[:, 2] >= qx0) & (bx[:, 0] <= qx1) & (bx[:, 3] >= qy0) & (bx[:, 1] <= qy1))[0]
    print(
        f"Open Buildings: {len(hit)} / {pf.metadata.num_row_groups} row groups intersect "
        f"the AOI (footer read {time.time() - t0:.0f}s)"
    )
    if len(hit) == 0:
        raise SystemExit("no Open Buildings row groups intersect the AOI")

    cols = ["bf_source", "confidence", "area_in_meters", "geometry", "bbox"]
    frames = []
    for rg in hit:
        tbl = pf.read_row_group(int(rg), columns=cols)
        bb = tbl.column("bbox").combine_chunks().flatten()
        xmin, ymin, xmax, ymax = (np.asarray(bb[i]) for i in range(4))
        keep = (xmax >= qx0) & (xmin <= qx1) & (ymax >= qy0) & (ymin <= qy1)
        if not keep.any():
            continue
        idx = np.where(keep)[0]
        sub = tbl.take(idx)
        frames.append(
            gpd.GeoDataFrame(
                {
                    "bf_source": np.asarray(sub.column("bf_source")).astype(str),
                    "confidence": np.asarray(sub.column("confidence"), dtype="float64"),
                    "area_in_meters": np.asarray(sub.column("area_in_meters"), dtype="float64"),
                },
                geometry=shapely.from_wkb(np.asarray(sub.column("geometry"))),
                crs="EPSG:4326",
            )
        )
    raw = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    by_source = raw["bf_source"].str.lower().value_counts().to_dict()

    # The mirror unions Google and Microsoft; Open Buildings v3 is the Google half.
    ob = raw[raw["bf_source"].str.lower() == "google"].to_crs(WORKING_CRS)
    ob = ob[ob.intersects(aoi.union_all())].reset_index(drop=True)
    ob["area_m2"] = ob.area
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    ob.to_file(out_gpkg, layer="footprints", driver="GPKG")

    provenance["footprints"] = {
        "product": "Google Open Buildings v3 (VIDA GeoParquet mirror, source.coop)",
        "url": OB_PARQUET,
        "license": "CC BY 4.0 / ODbL (Google Open Buildings)",
        "row_groups_read": [int(v) for v in hit],
        "row_groups_total": int(pf.metadata.num_row_groups),
        "candidates_in_bbox_by_source": by_source,
        "kept_google_in_aoi": int(len(ob)),
        "confidence_threshold": None,
        "confidence_note": "no threshold applied — this is the 'Open Buildings as-is' baseline",
        "confidence_quantiles": {
            q: float(ob["confidence"].quantile(float(q))) for q in ("0.05", "0.25", "0.5", "0.75")
        },
    }
    print(f"Footprints: {len(ob)} Google polygons in AOI (bbox candidates: {by_source})")
    return ob


def build_lod1(ob: gpd.GeoDataFrame, dtm_tif: Path, out_gpkg: Path, provenance: dict) -> None:
    """LoD1 prisms with base elevations sampled from GLO-30; heights left NULL.

    The height column exists so the schema mirrors the IPP target, but it stays
    NULL: fabricating a height would silently become a measured-looking number.
    """
    with rasterio.open(dtm_tif) as src:
        nodata = src.nodata
        # the DTM is kept on its native (geographic) grid, so sample in that CRS
        rp = gpd.GeoSeries(ob.geometry.representative_point(), crs=ob.crs).to_crs(src.crs)
        base = np.array([v[0] for v in src.sample([(p.x, p.y) for p in rp])], dtype="float64")
    if nodata is not None:
        base[base == nodata] = np.nan

    lod1 = ob[["confidence", "area_m2", "geometry"]].copy()
    lod1["base_elev_m"] = base
    lod1["height_m"] = np.nan
    lod1["roof_elev_m"] = np.nan
    lod1["height_source"] = "BLOCKED_GEE_AUTH"
    lod1.to_file(out_gpkg, layer="lod1", driver="GPKG")

    provenance["heights"] = {
        "product": "Open Buildings 2.5D Temporal (2019)",
        "status": "BLOCKED",
        "reason": HEIGHTS_BLOCKED_REASON,
        "probed_2026-09-08": [
            "gs://open-buildings-data (anonymous list) -> only v1/, v2/, v3/ polygon prefixes",
            "gs://open-buildings-temporal, open_buildings_temporal, open-buildings-2-5d -> 404",
        ],
        "effect": "height_m / roof_elev_m are NULL; no height metrics are computed",
    }
    n_base = int(np.isfinite(base).sum())
    print(f"LoD1: {len(lod1)} prisms, base elevation on {n_base}, height NULL (blocked)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--site-dir",
        type=Path,
        default=REPO / "outputs" / "comparative" / "satellite" / "rocinha",
    )
    args = ap.parse_args()

    aoi = gpd.read_file(args.site_dir / "aoi.gpkg", layer="aoi")
    recon = args.site_dir / "recon"
    recon.mkdir(parents=True, exist_ok=True)
    provenance: dict = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aoi": json.loads((args.site_dir / "aoi.json").read_text()),
        "leakage_rule": "no IPP ground-truth file is read by this pipeline",
    }

    dtm = fetch_dtm(aoi, recon / "dtm_glo30.tif", provenance)
    fetch_dtm(
        aoi,
        recon / "dtm_glo30_control.tif",
        provenance,
        buffer_m=CONTROL_BUFFER_M,
        key="dtm_control",
    )
    ob = fetch_footprints(aoi, recon / "footprints_ob_v3.gpkg", provenance)
    build_lod1(ob, dtm, recon / "lod1.gpkg", provenance)

    (recon / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"provenance -> {recon / 'provenance.json'}")


if __name__ == "__main__":
    main()
