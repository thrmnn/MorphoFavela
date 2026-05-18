#!/usr/bin/env python3
"""Build the VDG-P07 QGIS overlay project (no QGIS/PyQGIS required).

Generates, all in EPSG:31983 with repo-relative datasources:

  patches/VDG-P07/qgis/
    layers/analysis_disk.gpkg        Ø100 m analysis disk (50 m buffer of centre)
    layers/cfd_domain_qc.gpkg        axis-aligned analysis frame + 8 indicative
                                     per-direction rotated domain rectangles (QC)
    layers/patch_center.gpkg         centre point carrying every patch_meta field
    layers/terrain_hillshade.tif     gdaldem hillshade of the vendored terrain
    styles/*.qml                     standalone QGIS styles (Layer ▸ Load Style)
    VDG-P07.qgs / VDG-P07.qgz        project (styles embedded; zero re-styling)

The `CFD output (τ / ACH)` group is pre-styled (diverging ramp, legend, 60 %
opacity) and points at ../overlay/composite.tif — the Phase-3 deliverable —
so it renders the moment that file exists.

The 8 rotated domain rectangles are **indicative** (built from patch_meta +
the convention rot_ccw_deg = (90 + wind_deg) mod 360, cross-checked against
the RDP-P20 case_meta sample: wind 45° → rot 135°). Phase 3 consumes the
authoritative `rotation_rad_local_ccw` from each direction's case_meta.json
and regenerates this QC layer exactly; the attribute `indicative=1` marks it.

Run:  ~/miniconda3/envs/IVF/bin/python scripts/cfd_overlay/build_qgis_project.py \
          --patch VDG-P07 [--repo-root ~/IVF]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon

EPSG = 31983
WIND8 = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


def analysis_disk(cx: float, cy: float, r: float = 50.0) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"kind": ["analysis_disk"], "radius_m": [r]},
        geometry=[Point(cx, cy).buffer(r, quad_segs=128)],
        crs=EPSG,
    )


def domain_qc(cx, cy, meta) -> gpd.GeoDataFrame:
    """Axis-aligned analysis frame + 8 indicative rotated domain rectangles."""
    up = meta["domain_upstream_m"]
    down = meta["domain_downstream_m"]
    lat = meta["domain_lateral_m"]
    rows, geoms = [], []

    # Axis-aligned 100 m analysis frame (exact, deterministic).
    half = meta["analysis_patch_diameter"] / 2.0
    rows.append({"kind": "analysis_frame", "wind_deg": -1, "indicative": 0})
    geoms.append(Polygon([(cx - half, cy - half), (cx + half, cy - half),
                           (cx + half, cy + half), (cx - half, cy + half)]))

    # Per-direction domain rectangle: along-wind length = up+down (centre
    # offset asymmetrically by the upstream/downstream split), cross-wind
    # width = 2*lat. Local long axis = +x; rotate CCW about the centre.
    for d, wdeg in WIND8.items():
        base = Polygon([(-up, -lat), (down, -lat), (down, lat), (-up, lat)])
        base = translate(base, xoff=cx, yoff=cy)
        rot_ccw = (90.0 + wdeg) % 360.0
        geoms.append(rotate(base, rot_ccw, origin=(cx, cy), use_radians=False))
        rows.append({"kind": f"domain_{d}", "wind_deg": wdeg, "indicative": 1})

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=EPSG)


def patch_center(cx, cy, meta) -> gpd.GeoDataFrame:
    attrs = {k: [v] for k, v in meta.items()
             if isinstance(v, (int, float, str))}
    attrs["label"] = [
        f"{meta['patch_id']}  H̄={meta['H_mean']:.2f} m  "
        f"slope={meta['slope_deg']:.1f}°  λp={meta['lambda_p']:.3f}  "
        f"SVF={meta['svf']:.3f}  n={meta['n_buildings_in_domain']}"
    ]
    return gpd.GeoDataFrame(attrs, geometry=[Point(cx, cy)], crs=EPSG)


def hillshade(terrain: Path, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["gdaldem", "hillshade", "-az", "315", "-alt", "45",
             "-compute_edges", "-q", str(terrain), str(out)],
            check=True, capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ! hillshade skipped ({type(e).__name__}); "
              f"terrain layer still styled as elevation", file=sys.stderr)
        return False


# ---- QGIS style fragments (QML <qgis> subset; embedded in .qgs too) --------

def qml_raster_pseudocolor(band_min, band_max, ramp, opacity=1.0,
                           classes=12, label_suffix=""):
    stops = []
    for i in range(classes + 1):
        f = i / classes
        v = band_min + f * (band_max - band_min)
        r, g, b = ramp(f)
        stops.append((v, r, g, b))
    items = "".join(
        f'<item color="#{r:02x}{g:02x}{b:02x}" alpha="255" '
        f'value="{v:.6g}" label="{v:.3g}{label_suffix}"/>'
        for v, r, g, b in stops
    )
    return (
        f'<pipe><rasterrenderer type="singlebandpseudocolor" band="1" '
        f'opacity="{opacity}" classificationMin="{band_min:.6g}" '
        f'classificationMax="{band_max:.6g}">'
        f'<rastershader><colorrampshader colorRampType="INTERPOLATED" '
        f'clip="0">{items}</colorrampshader></rastershader>'
        f'</rasterrenderer><brightnesscontrast/><huesaturation/>'
        f'<rasterresampler/></pipe>'
    )


def qml_raster_gray(opacity=1.0):
    return (
        f'<pipe><rasterrenderer type="singlebandgray" band="1" '
        f'opacity="{opacity}" gradient="BlackToWhite"/>'
        f'<brightnesscontrast/><huesaturation/><rasterresampler/></pipe>'
    )


def ramp_terra(f):  # green→tan→brown elevation
    pts = [(0.0, (28, 102, 60)), (0.4, (190, 178, 110)),
           (0.75, (150, 96, 60)), (1.0, (240, 240, 240))]
    return _lerp(pts, f)


def ramp_div(f):  # diverging blue(low ACH/high τ, bad)→white→red(good)
    pts = [(0.0, (33, 102, 172)), (0.5, (247, 247, 247)),
           (1.0, (178, 24, 43))]
    return _lerp(pts, f)


def _lerp(pts, f):
    f = min(max(f, 0.0), 1.0)
    for (f0, c0), (f1, c1) in zip(pts, pts[1:]):
        if f <= f1:
            t = 0 if f1 == f0 else (f - f0) / (f1 - f0)
            return tuple(int(round(a + t * (b - a))) for a, b in zip(c0, c1))
    return pts[-1][1]


def qml_buildings():
    """Graduated by height; a dedicated red category for height==0 flags."""
    return (
        '<renderer-v2 type="RuleRenderer" symbollevels="0">'
        '<rules key="{r0}">'
        '<rule key="{r1}" filter="&quot;height&quot; = 0" '
        'label="height = 0 (flagged, n=5)" symbol="0"/>'
        '<rule key="{r2}" filter="&quot;height&quot; &gt; 0" '
        'label="footprint (graded by height)" symbol="1"/>'
        '</rules>'
        '<symbols>'
        '<symbol type="fill" name="0"><layer class="SimpleFill">'
        '<Option type="Map">'
        '<Option name="color" type="QString" value="227,26,28,255"/>'
        '<Option name="outline_color" type="QString" value="120,0,0,255"/>'
        '<Option name="outline_width" type="QString" value="0.4"/>'
        '</Option></layer></symbol>'
        '<symbol type="fill" name="1"><layer class="SimpleFill">'
        '<Option type="Map">'
        '<Option name="color" type="QString" value="140,150,160,180"/>'
        '<Option name="outline_color" type="QString" value="60,60,70,255"/>'
        '<Option name="outline_width" type="QString" value="0.15"/>'
        '</Option></layer></symbol>'
        '</symbols></renderer-v2>'
    ).format(r0="{00000000-0000-0000-0000-000000000000}",
             r1="{00000000-0000-0000-0000-000000000001}",
             r2="{00000000-0000-0000-0000-000000000002}")


def qml_outline(color, width, dashed=False):
    dash = ('<Option name="customdash" type="QString" value="4;2"/>'
            '<Option name="use_custom_dash" type="QString" value="1"/>'
            ) if dashed else ""
    return (
        '<renderer-v2 type="singleSymbol">'
        '<symbols><symbol type="fill" name="0"><layer class="SimpleFill">'
        '<Option type="Map">'
        '<Option name="style" type="QString" value="no"/>'
        f'<Option name="outline_color" type="QString" value="{color}"/>'
        f'<Option name="outline_width" type="QString" value="{width}"/>'
        f'{dash}</Option></layer></symbol></symbols></renderer-v2>'
    )


def qml_center_point():
    return (
        '<renderer-v2 type="singleSymbol">'
        '<symbols><symbol type="marker" name="0"><layer class="SimpleMarker">'
        '<Option type="Map">'
        '<Option name="name" type="QString" value="cross_fill"/>'
        '<Option name="color" type="QString" value="255,237,0,255"/>'
        '<Option name="outline_color" type="QString" value="0,0,0,255"/>'
        '<Option name="size" type="QString" value="4"/>'
        '</Option></layer></symbol></symbols></renderer-v2>'
        '<labeling type="simple"><settings><text-style fieldName="label" '
        'fontSize="9" namedStyle="Bold"><text-buffer bufferSize="1" '
        'bufferDraw="1"/></text-style><placement placement="1"/>'
        '</settings></labeling>'
    )


def build(repo_root: Path, patch: str):
    pdir = repo_root / "patches" / patch
    meta = json.loads((pdir / "inputs" / "patch_meta.json").read_text())
    cx, cy = meta["center_x"], meta["center_y"]
    qg = pdir / "qgis"
    (qg / "layers").mkdir(parents=True, exist_ok=True)
    (qg / "styles").mkdir(parents=True, exist_ok=True)

    analysis_disk(cx, cy).to_file(qg / "layers/analysis_disk.gpkg", driver="GPKG")
    domain_qc(cx, cy, meta).to_file(qg / "layers/cfd_domain_qc.gpkg", driver="GPKG")
    patch_center(cx, cy, meta).to_file(qg / "layers/patch_center.gpkg", driver="GPKG")
    terrain = pdir / "inputs" / "terrain.tif"
    have_hs = hillshade(terrain, qg / "layers/terrain_hillshade.tif")

    with rasterio.open(terrain) as r:
        arr = r.read(1, masked=True)
        tmin, tmax = float(arr.min()), float(arr.max())

    # layer spec: (tree-name, rel-source, provider, geomtype, style-xml)
    layers = []
    layers.append(("CFD output: τ / ACH composite (pending)",
                   "../overlay/composite.tif", "gdal", "raster",
                   qml_raster_pseudocolor(0, 1, ramp_div, opacity=0.6,
                                          label_suffix=" (rel)")))
    layers.append(("Patch centre + meta", "layers/patch_center.gpkg",
                   "ogr", "point", qml_center_point()))
    layers.append(("Ø100 m analysis disk", "layers/analysis_disk.gpkg",
                   "ogr", "polygon", qml_outline("255,127,0,255", 0.6)))
    layers.append(("CFD domain QC (indicative)", "layers/cfd_domain_qc.gpkg",
                   "ogr", "polygon", qml_outline("120,120,120,200", 0.3,
                                                  dashed=True)))
    layers.append(("Buildings (height; 0 flagged)",
                   "../inputs/buildings.gpkg", "ogr", "polygon", qml_buildings()))
    if have_hs:
        layers.append(("Terrain hillshade", "layers/terrain_hillshade.tif",
                       "gdal", "raster", qml_raster_gray(opacity=0.55)))
    layers.append(("Terrain elevation", "../inputs/terrain.tif",
                   "gdal", "raster",
                   qml_raster_pseudocolor(tmin, tmax, ramp_terra,
                                          label_suffix=" m")))

    # standalone .qml (Layer ▸ Load Style) — robust, version-tolerant
    for name, _src, _prov, _gt, style in layers:
        slug = name.split(":")[0].split("(")[0].strip().lower()
        slug = "".join(c if c.isalnum() else "_" for c in slug).strip("_")
        (qg / "styles" / f"{slug}.qml").write_text(
            f'<!DOCTYPE qgis><qgis version="3.28.0">{style}</qgis>\n')

    qgs = _qgs_xml(patch, layers)
    (qg / f"{patch}.qgs").write_text(qgs)
    ET.fromstring(qgs)  # structural self-check: well-formed XML or raises

    qgz = qg / f"{patch}.qgz"
    with zipfile.ZipFile(qgz, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{patch}.qgs", qgs)

    # ---- fail-closed structural checks --------------------------------
    problems = []
    for name, src, _prov, _gt, _s in layers:
        p = (qg / src).resolve()
        pending = src.endswith("composite.tif")
        if not p.exists() and not pending:
            problems.append(f"missing datasource: {src}")
    for lyr in (qg / "layers").glob("*.gpkg"):
        if gpd.read_file(lyr).crs.to_epsg() != EPSG:
            problems.append(f"{lyr.name} not EPSG:{EPSG}")
    if problems:
        print("STRUCTURAL CHECK FAILED:", *problems, sep="\n  ", file=sys.stderr)
        return 1

    print(f"OK  {qgz.relative_to(repo_root)}")
    print(f"    layers={len(layers)}  styles={len(layers)}  "
          f"hillshade={'yes' if have_hs else 'NO'}  crs=EPSG:{EPSG}")
    print(f"    composite.tif is the only pending datasource (Phase 3)")
    return 0


def _qgs_xml(patch: str, layers) -> str:
    srs = (
        '<spatialrefsys><authid>EPSG:31983</authid>'
        '<srsid>2363</srsid><srid>31983</srid><epsg>31983</epsg>'
        '<description>SIRGAS 2000 / UTM zone 23S</description>'
        '<projectionacronym>utm</projectionacronym>'
        '<ellipsoidacronym>EPSG:7019</ellipsoidacronym>'
        '<geographicflag>false</geographicflag></spatialrefsys>'
    )
    tree, maps, order = [], [], []
    for i, (name, src, prov, gt, style) in enumerate(layers):
        lid = f"L{i}_{patch.replace('-', '_')}"
        order.append(f'<item>{lid}</item>')
        tree.append(
            f'<layer-tree-layer id="{lid}" name="{name}" '
            f'source="{src}" providerKey="{prov}" '
            f'checked="Qt::Checked" expanded="0"/>'
        )
        if gt == "raster":
            body = (f'<provider>{prov}</provider>{style}'
                    f'<blendMode>0</blendMode>')
            ltype = 'type="raster"'
        else:
            wkb = {"point": "Point", "polygon": "Polygon"}[gt]
            body = (f'<provider>{prov}</provider>{style}'
                    f'<blendMode>0</blendMode>')
            ltype = f'type="vector" geometry="{wkb}"'
        maps.append(
            f'<maplayer {ltype} autoRefreshEnabled="0">'
            f'<id>{lid}</id><datasource>{src}</datasource>'
            f'<layername>{name}</layername>'
            f'<srs>{srs}</srs>'
            f'<provider>{prov}</provider>{body}</maplayer>'
        )
    return (
        '<!DOCTYPE qgis>'
        f'<qgis version="3.28.0-Firenze" projectname="{patch} — VDG overlay">'
        f'<projectCrs>{srs}</projectCrs>'
        '<layer-tree-group>'
        '<customproperties/>'
        f'{"".join(tree)}'
        '</layer-tree-group>'
        f'<layerorder>{"".join(order)}</layerorder>'
        f'<projectlayers>{"".join(maps)}</projectlayers>'
        '<properties><Gui><CanvasColor type="QString" value="255,255,255"/>'
        '</Gui></properties>'
        '</qgis>'
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="VDG-P07")
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parents[2])
    a = ap.parse_args()
    sys.exit(build(a.repo_root.resolve(), a.patch))
