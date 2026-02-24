"""Extract buildings/streets for a specific Rio favela from city-wide layers.

Example:
    /home/theo/miniconda3/envs/IVF/bin/python scripts/extract_favela_data.py \
        --favela "Rocinha"
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask


def normalize_text(value: str) -> str:
    """Normalize text for accent-insensitive matching."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.lower().strip()


def find_favela(favelas_gdf: gpd.GeoDataFrame, favela_name: str) -> gpd.GeoDataFrame:
    """Find favela(s) by case/accent-insensitive contains matching."""
    name_col = "nome" if "nome" in favelas_gdf.columns else "NOME"
    target = normalize_text(favela_name)
    normalized_names = favelas_gdf[name_col].astype(str).map(normalize_text)

    exact = favelas_gdf[normalized_names == target]
    if len(exact) > 0:
        return exact

    partial = favelas_gdf[normalized_names.str.contains(target, na=False)]
    return partial


def extract_favela_data(
    favela_name: str,
    data_rj_dir: Path,
    output_base_dir: Path,
    road_buffer_m: float = 50.0,
    terrain_buffer_m: float = 50.0,
    dtm_filename: str = "DTM_RJ.tif",
) -> dict:
    """Extract building/road layers for one favela."""
    favelas_path = data_rj_dir / "Favelas_Limit_2019.shp"
    buildings_path = data_rj_dir / "buildings_RJ_2019.shp"
    roads_path = data_rj_dir / "Logradouros.shp"
    dtm_path = data_rj_dir / dtm_filename

    for path in [favelas_path, buildings_path, roads_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
    if not dtm_path.exists():
        raise FileNotFoundError(f"Missing required terrain raster: {dtm_path}")

    print("Loading favela boundaries...")
    favelas_gdf = gpd.read_file(favelas_path)
    matches = find_favela(favelas_gdf, favela_name)
    if matches.empty:
        raise ValueError(f"Favela '{favela_name}' not found in {favelas_path.name}")

    if len(matches) > 1:
        names = matches["nome" if "nome" in matches.columns else "NOME"].tolist()
        raise ValueError(
            f"Favela name '{favela_name}' matched multiple records: {names}. "
            "Please use a more specific name."
        )

    favela_row = matches.iloc[[0]].copy()
    name_col = "nome" if "nome" in favela_row.columns else "NOME"
    resolved_name = str(favela_row.iloc[0][name_col])
    area_slug = normalize_text(resolved_name).replace(" ", "_")

    out_dir = output_base_dir / area_slug / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target favela: {resolved_name}")
    print(f"Output directory: {out_dir}")

    print("Loading city-wide buildings and roads...")
    buildings_gdf = gpd.read_file(buildings_path)
    roads_gdf = gpd.read_file(roads_path)

    if buildings_gdf.crs != favela_row.crs:
        buildings_gdf = buildings_gdf.to_crs(favela_row.crs)
    if roads_gdf.crs != favela_row.crs:
        roads_gdf = roads_gdf.to_crs(favela_row.crs)

    favela_geom = favela_row.geometry.iloc[0]
    favela_buffer_geom = favela_geom.buffer(road_buffer_m)

    # Buildings: centroid within favela boundary.
    print("Extracting buildings...")
    buildings_idx = list(buildings_gdf.sindex.intersection(favela_geom.bounds))
    buildings_subset = buildings_gdf.iloc[buildings_idx].copy()
    building_centroids = buildings_subset.geometry.centroid
    buildings_out = buildings_subset[building_centroids.within(favela_geom)].copy()

    # Roads: intersect favela + configurable buffer to keep near-edge connections.
    print("Extracting roads...")
    roads_idx = list(roads_gdf.sindex.intersection(favela_buffer_geom.bounds))
    roads_subset = roads_gdf.iloc[roads_idx].copy()
    roads_out = roads_subset[roads_subset.geometry.intersects(favela_buffer_geom)].copy()

    buildings_shp = out_dir / f"{area_slug}_buildings.shp"
    roads_shp = out_dir / f"roads_{area_slug}.shp"
    boundary_shp = out_dir / f"{area_slug}_boundary.shp"
    dtm_tif = out_dir / f"{area_slug}_dtm.tif"

    buildings_out.to_file(buildings_shp)
    roads_out.to_file(roads_shp)
    favela_row.to_file(boundary_shp)

    print("Extracting terrain (DTM)...")
    with rasterio.open(dtm_path) as src:
        favela_for_dtm = favela_row.to_crs(src.crs)
        terrain_geom = favela_for_dtm.geometry.iloc[0].buffer(terrain_buffer_m)
        out_image, out_transform = mask(src, [terrain_geom.__geo_interface__], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

    with rasterio.open(dtm_tif, "w", **out_meta) as dst:
        dst.write(out_image)

    readme_path = output_base_dir / area_slug / "README.md"
    readme_text = (
        f"# {resolved_name} Data\n\n"
        f"This directory contains extracted layers for **{resolved_name}**.\n\n"
        f"## Files\n"
        f"- `raw/{area_slug}_buildings.shp`: building footprints\n"
        f"- `raw/roads_{area_slug}.shp`: road centerlines (with {road_buffer_m:.0f}m boundary buffer)\n"
        f"- `raw/{area_slug}_boundary.shp`: favela boundary polygon\n"
        f"- `raw/{area_slug}_dtm.tif`: terrain raster clipped from `{dtm_filename}` "
        f"(with {terrain_buffer_m:.0f}m buffer)\n"
    )
    readme_path.write_text(readme_text, encoding="utf-8")

    result = {
        "favela_name": resolved_name,
        "area_slug": area_slug,
        "buildings_count": int(len(buildings_out)),
        "roads_count": int(len(roads_out)),
        "output_dir": str(out_dir),
        "buildings_path": str(buildings_shp),
        "roads_path": str(roads_shp),
        "boundary_path": str(boundary_shp),
        "dtm_path": str(dtm_tif),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract data for one Rio favela")
    parser.add_argument("--favela", required=True, help="Favela name (e.g., Rocinha)")
    parser.add_argument(
        "--data-rj-dir",
        default="/home/theo/IVF/data/RJ",
        help="Directory containing city-wide RJ shapefiles",
    )
    parser.add_argument(
        "--output-base-dir",
        default="/home/theo/IVF/data",
        help="Base directory for output area folders",
    )
    parser.add_argument(
        "--road-buffer-m",
        type=float,
        default=50.0,
        help="Buffer (meters) around boundary when extracting roads",
    )
    parser.add_argument(
        "--terrain-buffer-m",
        type=float,
        default=50.0,
        help="Buffer (meters) around boundary when clipping terrain raster",
    )
    parser.add_argument(
        "--dtm-filename",
        default="DTM_RJ.tif",
        help="Terrain raster filename inside data-rj-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = extract_favela_data(
        favela_name=args.favela,
        data_rj_dir=Path(args.data_rj_dir),
        output_base_dir=Path(args.output_base_dir),
        road_buffer_m=args.road_buffer_m,
        terrain_buffer_m=args.terrain_buffer_m,
        dtm_filename=args.dtm_filename,
    )
    print("\nExtraction complete:")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
