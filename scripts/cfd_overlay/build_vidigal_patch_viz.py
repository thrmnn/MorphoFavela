#!/usr/bin/env python3
"""Vidigal patch inspection viz — QGIS bundle + self-contained HTML.

Two deliverables, both under ``outputs/vidigal/dataviz/`` (gitignored —
they embed the large synthetic-CFD rasters, so they must NOT live in the
committed ``patches/`` tree alongside the lightweight VDG-P07 bundle):

  vidigal_patch_viz/                  self-contained QGIS folder
    layers/   all-22-patch disks+centres, VDG-P02 disk/domain/centre,
              TLS LoD2 + convex hull + disk∩hull overlap, P02 buildings
    rasters/  terrain + hillshade, synthetic U_mag (8 dirs + mean)
    styles/   *.qml
    vidigal_patches.qgs / .qgz       project (EPSG:31983, styles embedded)
    README.txt

  vidigal_patch_inspection.html       one Leaflet/folium file, layer
              groups: all patches · VDG-P02↔TLS overlap · synthetic U_mag
              (mean + 8 dirs). All geometry reprojected to EPSG:4326.

The synthetic field is the validated stand-in from
``scripts/generate_synthetic_cfd_results.py`` (provenance: synthetic=true).
It is NOT a CFD solve and is labelled as such in both deliverables.

Run:  python scripts/cfd_overlay/build_vidigal_patch_viz.py [--repo-root ~/IVF]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from branca.colormap import LinearColormap
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import Point

# reuse the tested single-patch QGIS generator's helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_qgis_project as bqp  # noqa: E402

EPSG = 31983
WIND8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
UMAG_VMAX = 3.0  # shared colour ceiling (m/s) so panels are comparable
LAWSON = 1.0  # pedestrian stagnation threshold (m/s)


# ---- pure, testable helpers ------------------------------------------------

def bin_umag_to_grid(x, y, u, spacing: float = 2.0):
    """Mean U_mag on an axis-aligned grid over the point bbox.

    Returns (arr, transform) where arr is float32 with NaN in empty
    cells and transform is the rasterio affine (north-up).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    u = np.asarray(u, float)
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    nx = int(np.floor((x1 - x0) / spacing)) + 1
    ny = int(np.floor((y1 - y0) / spacing)) + 1
    ix = np.clip(((x - x0) / spacing).astype(int), 0, nx - 1)
    iy = np.clip(((y1 - y) / spacing).astype(int), 0, ny - 1)  # row 0 = north
    flat = iy * nx + ix
    s = np.zeros(nx * ny)
    c = np.zeros(nx * ny)
    np.add.at(s, flat, u)
    np.add.at(c, flat, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / c, np.nan).reshape(ny, nx).astype(np.float32)
    transform = from_origin(x0 - spacing / 2, y1 + spacing / 2, spacing, spacing)
    return mean, transform


def overlap_pct(disk_geom, hull_geom) -> float:
    """Percent of the analysis disk covered by the hull."""
    if disk_geom.area == 0:
        return 0.0
    return float(disk_geom.intersection(hull_geom).area / disk_geom.area * 100.0)


# ---- raster io -------------------------------------------------------------

def write_geotiff(arr, transform, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32", crs=f"EPSG:{EPSG}", transform=transform,
        nodata=np.nan, compress="deflate",
    ) as dst:
        dst.write(arr, 1)


def _synthetic_umag(repo: Path, patch: str, direction: str):
    csv = repo / "data" / "vidigal" / "cfd_results_synthetic" / patch / direction / "sample_points.csv"
    d = pd.read_csv(csv, usecols=["x", "y", "U_mag"])
    return d["x"].to_numpy(), d["y"].to_numpy(), d["U_mag"].to_numpy()


def _rgba_from_grid(arr, vmax=UMAG_VMAX):
    """Turbo-coloured RGBA uint8 with transparent NaN cells."""
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    rgba = cm.turbo(norm(np.ma.masked_invalid(arr)))
    rgba = (rgba * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isnan(arr), 0, 200).astype(np.uint8)
    return rgba


def _reproject_to_4326(arr, transform):
    """Warp a EPSG:31983 grid to EPSG:4326; return (arr4326, [[S,W],[N,E]])."""
    src_crs = CRS.from_epsg(EPSG)
    dst_crs = CRS.from_epsg(4326)
    h, w = arr.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    dt, dw, dh = calculate_default_transform(
        src_crs, dst_crs, w, h, left=left, bottom=bottom, right=right, top=top
    )
    out = np.full((dh, dw), np.nan, np.float32)
    reproject(
        arr, out, src_transform=transform, src_crs=src_crs,
        dst_transform=dt, dst_crs=dst_crs, resampling=Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan,
    )
    w_lon, n_lat = dt * (0, 0)
    e_lon, s_lat = dt * (dw, dh)
    return out, [[s_lat, w_lon], [n_lat, e_lon]]


# ---- layer construction ----------------------------------------------------

def all_patch_disks(camp: pd.DataFrame) -> gpd.GeoDataFrame:
    rows, geoms = [], []
    for _, r in camp.iterrows():
        focus = r.patch_id in ("VDG-P02", "VDG-P07")
        rows.append({
            "patch_id": r.patch_id, "stratum_id": r.stratum_id,
            "svf": round(float(r.svf), 4), "lambda_p": round(float(r.lambda_p), 4),
            "slope_deg": round(float(r.slope_deg), 2),
            "H_mean": round(float(r.H_mean), 2),
            "is_focus": int(focus),
            "label": (f"{r.patch_id}  SVF={r.svf:.3f} λp={r.lambda_p:.3f} "
                      f"slope={r.slope_deg:.1f}°"),
        })
        geoms.append(Point(r.center_x, r.center_y).buffer(50.0, quad_segs=96))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=EPSG)


def tls_layers(repo: Path, p02_disk):
    lod2 = gpd.read_file(repo / "data" / "vidigal_tls" / "raw" / "vidigal_LoD2.gpkg")
    lod2 = lod2.to_crs(EPSG)
    hull = lod2.geometry.union_all().convex_hull
    hull_gdf = gpd.GeoDataFrame({"kind": ["tls_lod2_convex_hull"]},
                                geometry=[hull], crs=EPSG)
    pct = overlap_pct(p02_disk, hull)
    inter = p02_disk.intersection(hull)
    ov_gdf = gpd.GeoDataFrame(
        {"kind": ["vdgp02_disk_in_tls"], "overlap_pct": [round(pct, 1)]},
        geometry=[inter], crs=EPSG,
    )
    return lod2, hull_gdf, ov_gdf, pct


# ---- QGIS bundle -----------------------------------------------------------

def umag_ramp(f):  # blue(calm/bad)→cyan→yellow→red(windy) — turbo-ish
    pts = [(0.0, (48, 18, 59)), (0.25, (33, 144, 230)), (0.5, (60, 200, 120)),
           (0.75, (240, 200, 40)), (1.0, (180, 24, 43))]
    return bqp._lerp(pts, f)


def build_qgis(repo: Path, out: Path, camp, meta, p02_disk_gdf,
               lod2, hull_gdf, ov_gdf, pct) -> int:
    qg = out / "qgis"
    (qg / "layers").mkdir(parents=True, exist_ok=True)
    (qg / "rasters").mkdir(parents=True, exist_ok=True)
    (qg / "styles").mkdir(parents=True, exist_ok=True)
    cx, cy = meta["center_x"], meta["center_y"]

    all_patch_disks(camp).to_file(qg / "layers/all_patch_disks.gpkg", driver="GPKG")
    bqp.patch_center(cx, cy, meta).to_file(qg / "layers/vdgp02_center.gpkg", driver="GPKG")
    p02_disk_gdf.to_file(qg / "layers/vdgp02_analysis_disk.gpkg", driver="GPKG")
    bqp.domain_qc(cx, cy, meta).to_file(qg / "layers/vdgp02_cfd_domain_qc.gpkg", driver="GPKG")
    lod2.to_file(qg / "layers/tls_lod2.gpkg", driver="GPKG")
    hull_gdf.to_file(qg / "layers/tls_hull.gpkg", driver="GPKG")
    ov_gdf.to_file(qg / "layers/vdgp02_tls_overlap.gpkg", driver="GPKG")
    gpd.read_file(repo / "patches/VDG-P02/inputs/buildings.gpkg").to_crs(EPSG).to_file(
        qg / "layers/vdgp02_buildings.gpkg", driver="GPKG")

    # copy terrain into the bundle so the folder is fully portable
    terrain = repo / "patches/VDG-P02/inputs/terrain.tif"
    import shutil
    shutil.copyfile(terrain, qg / "rasters/vdgp02_terrain.tif")
    have_hs = bqp.hillshade(terrain, qg / "rasters/vdgp02_terrain_hillshade.tif")

    umag_min, umag_max = 0.0, UMAG_VMAX
    stack = []
    for d in WIND8:
        x, y, u = _synthetic_umag(repo, "VDG-P02", d)
        arr, tr = bin_umag_to_grid(x, y, u)
        write_geotiff(arr, tr, qg / f"rasters/vdgp02_synthetic_Umag_{d}.tif")
        stack.append((arr, tr))
    # 8-dir mean over the union (cells averaged where covered)
    xs = np.concatenate([_synthetic_umag(repo, "VDG-P02", d)[0] for d in WIND8])
    ys = np.concatenate([_synthetic_umag(repo, "VDG-P02", d)[1] for d in WIND8])
    us = np.concatenate([_synthetic_umag(repo, "VDG-P02", d)[2] for d in WIND8])
    mean_arr, mean_tr = bin_umag_to_grid(xs, ys, us)
    write_geotiff(mean_arr, mean_tr, qg / "rasters/vdgp02_synthetic_Umag_mean.tif")

    umag_style = bqp.qml_raster_pseudocolor(
        umag_min, umag_max, umag_ramp, opacity=0.75, label_suffix=" m/s")
    with rasterio.open(terrain) as r:
        a = r.read(1, masked=True)
        tmin, tmax = float(a.min()), float(a.max())

    layers = []
    layers.append(("Synthetic U_mag — 8-dir mean (SYNTHETIC)",
                   "qgis/rasters/vdgp02_synthetic_Umag_mean.tif",
                   "gdal", "raster", umag_style))
    for d in WIND8:
        layers.append((f"Synthetic U_mag — {d} (SYNTHETIC)",
                       f"qgis/rasters/vdgp02_synthetic_Umag_{d}.tif",
                       "gdal", "raster", umag_style))
    layers.append(("VDG-P02 ∩ TLS overlap",
                   "qgis/layers/vdgp02_tls_overlap.gpkg", "ogr", "polygon",
                   bqp.qml_outline("0,160,80,255", 0.8)))
    layers.append(("TLS LoD2 convex hull",
                   "qgis/layers/tls_hull.gpkg", "ogr", "polygon",
                   bqp.qml_outline("0,120,200,255", 0.6, dashed=True)))
    layers.append(("TLS LoD2 footprints",
                   "qgis/layers/tls_lod2.gpkg", "ogr", "polygon",
                   bqp.qml_outline("0,120,200,180", 0.25)))
    layers.append(("VDG-P02 buildings (height; 0 flagged)",
                   "qgis/layers/vdgp02_buildings.gpkg", "ogr", "polygon",
                   bqp.qml_buildings()))
    layers.append(("VDG-P02 centre + meta",
                   "qgis/layers/vdgp02_center.gpkg", "ogr", "point",
                   bqp.qml_center_point()))
    layers.append(("VDG-P02 Ø100 m analysis disk",
                   "qgis/layers/vdgp02_analysis_disk.gpkg", "ogr", "polygon",
                   bqp.qml_outline("255,127,0,255", 0.7)))
    layers.append(("VDG-P02 CFD domain QC (indicative)",
                   "qgis/layers/vdgp02_cfd_domain_qc.gpkg", "ogr", "polygon",
                   bqp.qml_outline("120,120,120,200", 0.3, dashed=True)))
    layers.append(("All 22 vidigal patch disks",
                   "qgis/layers/all_patch_disks.gpkg", "ogr", "polygon",
                   bqp.qml_outline("150,90,160,220", 0.4)))
    if have_hs:
        layers.append(("Terrain hillshade",
                       "qgis/rasters/vdgp02_terrain_hillshade.tif",
                       "gdal", "raster", bqp.qml_raster_gray(opacity=0.55)))
    layers.append(("Terrain elevation", "qgis/rasters/vdgp02_terrain.tif",
                   "gdal", "raster",
                   bqp.qml_raster_pseudocolor(tmin, tmax, bqp.ramp_terra,
                                              label_suffix=" m")))

    for name, _s, _p, _g, style in layers:
        slug = "".join(c if c.isalnum() else "_"
                       for c in name.split("(")[0].strip().lower()).strip("_")
        (qg / "styles" / f"{slug}.qml").write_text(
            f'<!DOCTYPE qgis><qgis version="3.28.0">{style}</qgis>\n')

    # .qgs/.qgz live at the bundle root; every datasource is "qgis/..."
    # relative to it → the folder is fully portable, no repo coupling.
    qgs = bqp._qgs_xml("vidigal_patches", layers)
    (out / "vidigal_patches.qgs").write_text(qgs)
    import xml.etree.ElementTree as ET
    import zipfile
    ET.fromstring(qgs)
    with zipfile.ZipFile(out / "vidigal_patches.qgz", "w",
                         zipfile.ZIP_DEFLATED) as z:
        z.writestr("vidigal_patches.qgs", qgs)

    (out / "README.txt").write_text(
        "Vidigal patch inspection — QGIS bundle\n"
        "======================================\n\n"
        "Open vidigal_patches.qgz in QGIS (EPSG:31983, styles embedded).\n"
        "Datasources are relative; keep this folder inside the IVF repo.\n\n"
        f"VDG-P02 ∩ TLS LoD2 convex hull = {pct:.1f}% of the Ø100 m disk\n"
        "(provenance figure: 66.6%).\n\n"
        "WARNING: the 'Synthetic U_mag' rasters are NOT a CFD solve. They\n"
        "come from scripts/generate_synthetic_cfd_results.py (summary.json\n"
        "carries synthetic:true + provenance) and exist for dataviz /\n"
        "figure-format prototyping only. Do not cite as results.\n")

    miss = [s for (_n, s, _p, _g, _st) in layers if not (out / s).exists()]
    if miss:
        print("STRUCTURAL CHECK FAILED (missing datasources):",
              *miss, sep="\n  ", file=sys.stderr)
        return 1
    print(f"OK  {(out / 'vidigal_patches.qgz').relative_to(repo)}")
    print(f"    layers={len(layers)}  hillshade={'yes' if have_hs else 'NO'}  "
          f"overlap={pct:.1f}%  crs=EPSG:{EPSG}")
    return 0


# ---- HTML ------------------------------------------------------------------

def _img_overlay(arr, transform, name, show):
    # warp the float field first, then recolour, so bilinear resampling
    # never blends across the transparent NaN edge in RGBA space
    warped, bounds = _reproject_to_4326(arr, transform)
    return folium.raster_layers.ImageOverlay(
        image=_rgba_from_grid(warped), bounds=bounds, opacity=0.78,
        name=name, show=show, interactive=False, cross_origin=False,
    )


def build_html(repo: Path, out_html: Path, camp, meta, p02_disk_gdf,
               lod2, hull_gdf, ov_gdf, pct) -> int:
    def to4326(g):
        return g.to_crs(4326)

    cx, cy = meta["center_x"], meta["center_y"]
    centre4326 = gpd.GeoSeries([Point(cx, cy)], crs=EPSG).to_crs(4326).iloc[0]

    m = folium.Map(location=[centre4326.y, centre4326.x], zoom_start=16,
                    tiles="CartoDB positron", control_scale=True)

    # --- group 1: all 22 patch disks
    g_all = folium.FeatureGroup(name="All 22 vidigal patch disks", show=True)
    disks = to4326(all_patch_disks(camp))
    for _, r in disks.iterrows():
        focus = bool(r.is_focus)
        folium.GeoJson(
            r.geometry,
            style_function=(lambda _f, fo=focus: {
                "color": "#d55e00" if fo else "#8c5aa0",
                "weight": 3 if fo else 1.5,
                "fillOpacity": 0.18 if fo else 0.06}),
            tooltip=r.label,
        ).add_to(g_all)
    g_all.add_to(m)

    # --- group 2: VDG-P02 ↔ TLS overlap
    g_tls = folium.FeatureGroup(name="VDG-P02 ↔ TLS overlap", show=True)
    folium.GeoJson(to4326(gpd.read_file(
        repo / "patches/VDG-P02/inputs/buildings.gpkg").to_crs(EPSG)),
        style_function=lambda _f: {"color": "#555", "weight": 0.3,
                                   "fillColor": "#888", "fillOpacity": 0.35},
        name="VDG-P02 buildings").add_to(g_tls)
    folium.GeoJson(to4326(lod2),
        style_function=lambda _f: {"color": "#0078c8", "weight": 0.5,
                                   "fillColor": "#0078c8", "fillOpacity": 0.12},
        name="TLS LoD2 footprints").add_to(g_tls)
    folium.GeoJson(to4326(hull_gdf),
        style_function=lambda _f: {"color": "#0078c8", "weight": 2,
                                   "dashArray": "6 4", "fill": False},
        tooltip="TLS LoD2 convex hull").add_to(g_tls)
    folium.GeoJson(to4326(p02_disk_gdf),
        style_function=lambda _f: {"color": "#ff7f00", "weight": 3,
                                   "fill": False},
        tooltip="VDG-P02 Ø100 m analysis disk").add_to(g_tls)
    folium.GeoJson(to4326(ov_gdf),
        style_function=lambda _f: {"color": "#00a050", "weight": 2,
                                   "fillColor": "#00d070", "fillOpacity": 0.45},
        tooltip=f"VDG-P02 ∩ TLS = {pct:.1f}% of the analysis disk"
        ).add_to(g_tls)
    folium.Marker(
        [centre4326.y, centre4326.x],
        icon=folium.DivIcon(html=(
            '<div style="font:600 11px sans-serif;color:#b00;'
            'background:#fff;border:1px solid #b00;padding:1px 4px;'
            f'border-radius:3px">VDG-P02 — TLS overlap {pct:.1f}%</div>')),
    ).add_to(g_tls)
    g_tls.add_to(m)

    # --- group 3: synthetic U_mag (mean shown; 8 dirs available)
    x, y, u = (np.concatenate([_synthetic_umag(repo, "VDG-P02", d)[i]
               for d in WIND8]) for i in range(3))
    mean_arr, mean_tr = bin_umag_to_grid(x, y, u)
    _img_overlay(mean_arr, mean_tr, "Synthetic U_mag — 8-dir mean (SYNTHETIC)",
                 True).add_to(m)
    for d in WIND8:
        xx, yy, uu = _synthetic_umag(repo, "VDG-P02", d)
        a, tr = bin_umag_to_grid(xx, yy, uu)
        _img_overlay(a, tr, f"Synthetic U_mag — {d} (SYNTHETIC)",
                     False).add_to(m)

    cmap = LinearColormap(
        [to_hex(cm.turbo(v)) for v in np.linspace(0, 1, 12)],
        vmin=0, vmax=UMAG_VMAX,
        caption=f"Synthetic pedestrian U_mag (m/s) — Lawson stagnation = {LAWSON:g} m/s",
    )
    cmap.add_to(m)

    banner = folium.Element(
        '<div style="position:fixed;top:8px;left:50%;transform:translateX(-50%);'
        'z-index:9999;background:#fff3cd;border:1px solid #c79100;'
        'padding:4px 10px;font:600 12px sans-serif;border-radius:4px">'
        'Vidigal patch inspection — VDG-P02 ↔ TLS · '
        '<span style="color:#b00">SYNTHETIC U_mag (not a CFD solve)</span>'
        '</div>')
    m.get_root().html.add_child(banner)
    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(f"OK  {out_html.relative_to(repo)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parents[2])
    a = ap.parse_args()
    repo = a.repo_root.resolve()

    import json
    meta = json.loads((repo / "patches/VDG-P02/inputs/patch_meta.json").read_text())
    camp = pd.read_csv(
        repo / "outputs/vidigal/sampling_cfd/campaign_sampling/campaign_patches.csv")
    p02_disk_gdf = bqp.analysis_disk(meta["center_x"], meta["center_y"])
    lod2, hull_gdf, ov_gdf, pct = tls_layers(repo, p02_disk_gdf.geometry.iloc[0])
    if abs(pct - 66.6) > 5.0:
        print(f"  ! TLS overlap {pct:.1f}% deviates from provenance 66.6% "
              f"(hull definition may differ)", file=sys.stderr)

    out = repo / "outputs/vidigal/dataviz/vidigal_patch_viz"
    rc = build_qgis(repo, out, camp, meta, p02_disk_gdf,
                    lod2, hull_gdf, ov_gdf, pct)
    rc |= build_html(repo, repo / "outputs/vidigal/dataviz/vidigal_patch_inspection.html",
                     camp, meta, p02_disk_gdf, lod2, hull_gdf, ov_gdf, pct)
    return rc


if __name__ == "__main__":
    sys.exit(main())
