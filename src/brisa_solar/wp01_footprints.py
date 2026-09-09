"""WP-01: normalise the citywide footprint layer to a single CRS.

`data/RJ/buildings_RJ_2019.shp` declares EPSG:31983 (SIRGAS 2000 / UTM 23S) but
roughly one feature in five is stored in GEOGRAPHIC degrees instead of projected
metres. Measured 2026-09-08 on a systematic 1-in-200 sample (11,815 of
2,362,806): 2,337 features (19.8%) carried coordinates near x=-43, y=-23.

Proof they are the same data and not a separate population: reprojecting them
4674 -> 31983 puts 100% inside the DTM's bounds, and their `base` elevation then
agrees with the DTM to a median 0.166 m — indistinguishable from the 0.158 m of
the already-projected majority.

Left unrepaired, every citywide computation (WP-05's domain, density classes,
coverage thresholds) silently runs on ~80% of the city's buildings, because the
mis-stored fifth falls outside every projected raster and is dropped.

Detection is per-feature and geometric, never per-file: a feature is treated as
geographic when its coordinates fall inside the valid degree envelope, which no
UTM 23S coordinate for Rio can do (eastings are ~10^5, northings ~10^6).

Run: python3 -m src.brisa_solar.wp01_footprints [--out PATH] [--limit N]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import fiona
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform

from .constants import REPO_ROOT, load_params

#: SIRGAS 2000 geographic — the geographic twin of EPSG:31983.
GEOGRAPHIC_EPSG = 4674
PROJECTED_EPSG = 31983

_TRANSFORMER = Transformer.from_crs(f"EPSG:{GEOGRAPHIC_EPSG}", f"EPSG:{PROJECTED_EPSG}", always_xy=True)


def looks_geographic(x: float, y: float) -> bool:
    """True when a coordinate pair is degrees rather than UTM 23S metres.

    Unambiguous here: a Rio UTM easting is ~6.2-7.0e5 and a northing ~7.4e6, so
    nothing legitimate lands inside the degree envelope.
    """
    return abs(x) <= 180.0 and abs(y) <= 90.0


def normalise(src_path: Path, out_path: Path, limit: int | None = None) -> dict:
    reprojected = failed = passthrough = 0
    with fiona.open(src_path) as src:
        meta = src.meta.copy()
        meta["crs"] = f"EPSG:{PROJECTED_EPSG}"
        meta["crs_wkt"] = ""
        # Driver follows the requested extension, not the source's: inheriting
        # "ESRI Shapefile" for a .gpkg path silently writes a shapefile into a
        # directory named *.gpkg, and shapefiles cap at 2 GB.
        meta["driver"] = "GPKG" if out_path.suffix.lower() == ".gpkg" else meta.get("driver", "GPKG")
        if meta["driver"] == "GPKG":
            meta.pop("crs_wkt", None)
        # The source mixes Polygon and MultiPolygon under a '3D Polygon' schema,
        # so a typed output schema rejects records mid-write. "Unknown" accepts
        # both. Geometry Z is also dropped here: the obstruction surface takes
        # its heights from the base/altura/topo attributes, never from geometry
        # Z, so this loses nothing the pipeline uses — but it IS a change, and
        # is reported rather than left implicit.
        schema = dict(meta["schema"])
        schema["geometry"] = "Unknown"
        meta["schema"] = schema
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with fiona.open(out_path, "w", **meta) as dst:
            for i, feat in enumerate(src):
                if limit is not None and i >= limit:
                    break
                geom = shape(feat["geometry"])
                if geom.is_empty:
                    failed += 1
                    continue
                c = geom.representative_point()
                if looks_geographic(c.x, c.y):
                    geom = shapely_transform(
                        lambda xx, yy, zz=None: _TRANSFORMER.transform(xx, yy), geom
                    )
                    reprojected += 1
                else:
                    passthrough += 1
                dst.write({"geometry": mapping(geom), "properties": feat["properties"]})

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    return {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": rel(src_path),
        "output": rel(out_path),
        "declared_crs": f"EPSG:{PROJECTED_EPSG}",
        "features_passthrough_already_projected": passthrough,
        "features_reprojected_from_geographic": reprojected,
        "features_dropped_empty_geometry": failed,
        "fraction_reprojected": round(reprojected / max(reprojected + passthrough, 1), 4),
        "geometry_note": "Output is 2D; source 3D Polygon/MultiPolygon mix written as Unknown geometry type. Heights come from base/altura/topo attributes, never geometry Z.",
        "geographic_crs_assumed": f"EPSG:{GEOGRAPHIC_EPSG} (SIRGAS 2000 geographic; EPSG:4326 gives an "
                                  "identical result at this precision — the datums agree to ~1 m)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/RJ/buildings_RJ_2019_utm.gpkg")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    params = load_params()
    src = REPO_ROOT / params["footprints"]["path"]
    out = REPO_ROOT / args.out
    report = normalise(src, out, args.limit)

    run_dir = REPO_ROOT / "runs" / datetime.now(timezone.utc).strftime("wp01_footprints_%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "normalisation_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
