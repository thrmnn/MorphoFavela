"""Export the Rocinha AOI polygon for the free-EO reconstruction.

This is the ONLY step that reads the municipal limits layer. The AOI answers
"where is Rocinha", which is not part of the answer key — the answer key is the
IPP terrain and building data, and those are quarantined from the
reconstruction (see `build_reconstruction.py`). Materialising the AOI here lets
the reconstruction run against a single neutral file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd

REPO = Path(__file__).resolve().parents[2]
LIMITS = REPO / "data" / "RJ" / "Favelas_Limit_2019.shp"
# Rocinha proper. The name matches three polygons in the layer (Rocinha,
# "Matinha (RA - Rocinha)", "Vila Parque da Cidade"); the code is unambiguous.
ROCINHA_COD_FAVELA = 43
WORKING_CRS = "EPSG:31983"


def export_aoi(out_dir: Path, cod_favela: int = ROCINHA_COD_FAVELA) -> Path:
    limits = gpd.read_file(LIMITS)
    aoi = limits[limits["cod_favela"] == cod_favela].to_crs(WORKING_CRS)
    if len(aoi) != 1:
        raise SystemExit(f"expected exactly 1 polygon for cod_favela={cod_favela}, got {len(aoi)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "aoi.gpkg"
    aoi[["cod_favela", "nome", "geometry"]].to_file(gpkg, layer="aoi", driver="GPKG")

    wgs = aoi.to_crs("EPSG:4326")
    meta = {
        "site": str(aoi.iloc[0]["nome"]),
        "cod_favela": int(cod_favela),
        "source_layer": str(LIMITS.relative_to(REPO)),
        "crs": WORKING_CRS,
        "area_m2": float(aoi.area.sum()),
        "bounds_31983": [float(v) for v in aoi.total_bounds],
        "bounds_4326": [float(v) for v in wgs.total_bounds],
    }
    (out_dir / "aoi.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"AOI {meta['site']}: {meta['area_m2'] / 1e6:.3f} km2 -> {gpkg}")
    return gpkg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs" / "comparative" / "satellite" / "rocinha",
    )
    ap.add_argument("--cod-favela", type=int, default=ROCINHA_COD_FAVELA)
    args = ap.parse_args()
    export_aoi(args.out_dir, args.cod_favela)


if __name__ == "__main__":
    main()
