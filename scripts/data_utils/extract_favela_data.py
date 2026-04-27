"""Extract buildings/streets for a specific Rio favela from city-wide layers.

Examples:
    # Single favela by name:
    python scripts/extract_favela_data.py --favela "Rocinha"

    # Complex of sub-favelas (dissolved into single boundary):
    python scripts/extract_favela_data.py --complexo "Complexo do Alemão" --dissolve
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.ops import unary_union


def normalize_text(value: str) -> str:
    """Normalize text for accent-insensitive matching."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.lower().strip()


def find_favela(favelas_gdf: gpd.GeoDataFrame, favela_name: str) -> gpd.GeoDataFrame:
    """Find favela(s) by case/accent-insensitive contains matching on the `nome` column."""
    name_col = "nome" if "nome" in favelas_gdf.columns else "NOME"
    target = normalize_text(favela_name)
    normalized_names = favelas_gdf[name_col].astype(str).map(normalize_text)

    exact = favelas_gdf[normalized_names == target]
    if len(exact) > 0:
        return exact

    partial = favelas_gdf[normalized_names.str.contains(target, na=False)]
    return partial


def find_complex(favelas_gdf: gpd.GeoDataFrame, complex_name: str) -> gpd.GeoDataFrame:
    """Find sub-favelas by case/accent-insensitive matching on the `complexo` column."""
    col = "complexo" if "complexo" in favelas_gdf.columns else "COMPLEXO"
    target = normalize_text(complex_name)
    normalized = favelas_gdf[col].astype(str).map(normalize_text)

    exact = favelas_gdf[normalized == target]
    if len(exact) > 0:
        return exact

    partial = favelas_gdf[normalized.str.contains(target, na=False)]
    return partial


def _fix_mixed_crs(
    gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:31983"
) -> gpd.GeoDataFrame:
    """Detect and fix datasets with mixed CRS (some rows in lat/lon, rest projected).

    The RJ buildings shapefile contains ~467k rows stored in WGS84 (EPSG:4326)
    coordinates while the remaining ~1.9M are in EPSG:31983.  The file header
    declares EPSG:31983, so GeoPandas does not reproject the lat/lon rows.

    This function splits the GeoDataFrame by a simple heuristic (x < 1000 means
    lat/lon), reprojects the lat/lon partition, and concatenates the result.
    """
    bounds = gdf.geometry.bounds
    latlon_mask = bounds["minx"] < 1000  # degrees vs hundreds-of-thousands of metres
    n_latlon = int(latlon_mask.sum())
    if n_latlon == 0:
        # All rows already in projected CRS — nothing to fix.
        return gdf

    print(
        f"  Mixed-CRS detected: {n_latlon} rows in lat/lon, "
        f"{len(gdf) - n_latlon} in projected. Reprojecting lat/lon rows..."
    )

    projected_part = gdf[~latlon_mask].copy()
    latlon_part = gdf[latlon_mask].copy()

    # Temporarily set CRS to WGS84 for the lat/lon partition, then reproject.
    latlon_part = latlon_part.set_crs("EPSG:4326", allow_override=True)
    latlon_part = latlon_part.to_crs(target_crs)

    return pd.concat([projected_part, latlon_part], ignore_index=True)


def extract_favela_data(
    favela_name: str,
    data_rj_dir: Path,
    output_base_dir: Path,
    road_buffer_m: float = 50.0,
    terrain_buffer_m: float = 50.0,
    dtm_filename: str = "DTM_RJ.tif",
    boundary_gdf: gpd.GeoDataFrame | None = None,
    area_slug: str | None = None,
    display_name: str | None = None,
) -> dict:
    """Extract building/road layers for one favela or complex.

    Parameters
    ----------
    favela_name : str
        Name used for lookup when *boundary_gdf* is not provided.
    data_rj_dir : Path
        Directory containing city-wide RJ shapefiles.
    output_base_dir : Path
        Base directory for output area folders.
    road_buffer_m : float
        Buffer (metres) around boundary when extracting roads.
    terrain_buffer_m : float
        Buffer (metres) around boundary when clipping terrain raster.
    dtm_filename : str
        Terrain raster filename inside *data_rj_dir*.
    boundary_gdf : GeoDataFrame, optional
        Pre-provided boundary (e.g. dissolved complex). When ``None``,
        the function falls back to the ``find_favela()`` lookup on *favela_name*.
    area_slug : str, optional
        Override for the output folder / filename slug.
    display_name : str, optional
        Override for the human-readable name used in logs and README.
    """
    favelas_path = data_rj_dir / "Favelas_Limit_2019.shp"
    buildings_path = data_rj_dir / "buildings_RJ_2019.shp"
    roads_path = data_rj_dir / "Logradouros.shp"
    dtm_path = data_rj_dir / dtm_filename

    for path in [buildings_path, roads_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
    if not dtm_path.exists():
        raise FileNotFoundError(f"Missing required terrain raster: {dtm_path}")

    # ---- Resolve boundary ------------------------------------------------
    if boundary_gdf is not None:
        favela_row = boundary_gdf.copy()
        resolved_name = display_name or favela_name
        slug = area_slug or normalize_text(resolved_name).replace(" ", "_")
    else:
        if not favelas_path.exists():
            raise FileNotFoundError(f"Missing required file: {favelas_path}")
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
        resolved_name = display_name or str(favela_row.iloc[0][name_col])
        slug = area_slug or normalize_text(resolved_name).replace(" ", "_")

    out_dir = output_base_dir / slug / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target area: {resolved_name}")
    print(f"Output directory: {out_dir}")

    target_crs = str(favela_row.crs)

    print("Loading city-wide buildings...")
    buildings_gdf = gpd.read_file(buildings_path)
    buildings_gdf = _fix_mixed_crs(buildings_gdf, target_crs)
    if buildings_gdf.crs != favela_row.crs:
        buildings_gdf = buildings_gdf.to_crs(favela_row.crs)

    print("Loading city-wide roads...")
    roads_gdf = gpd.read_file(roads_path)
    roads_gdf = _fix_mixed_crs(roads_gdf, target_crs)
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
    roads_out = roads_subset[
        roads_subset.geometry.intersects(favela_buffer_geom)
    ].copy()

    buildings_shp = out_dir / f"{slug}_buildings.shp"
    roads_shp = out_dir / f"roads_{slug}.shp"
    boundary_shp = out_dir / f"{slug}_boundary.shp"
    dtm_tif = out_dir / f"{slug}_dtm.tif"

    buildings_out.to_file(buildings_shp)
    roads_out.to_file(roads_shp)
    favela_row.to_file(boundary_shp)

    print("Extracting terrain (DTM)...")
    with rasterio.open(dtm_path) as src:
        favela_for_dtm = favela_row.to_crs(src.crs)
        terrain_geom = favela_for_dtm.geometry.iloc[0].buffer(terrain_buffer_m)
        out_image, out_transform = mask(
            src, [terrain_geom.__geo_interface__], crop=True
        )
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

    readme_path = output_base_dir / slug / "README.md"
    readme_text = (
        f"# {resolved_name} Data\n\n"
        f"This directory contains extracted layers for **{resolved_name}**.\n\n"
        f"## Files\n"
        f"- `raw/{slug}_buildings.shp`: building footprints\n"
        f"- `raw/roads_{slug}.shp`: road centerlines (with {road_buffer_m:.0f}m boundary buffer)\n"
        f"- `raw/{slug}_boundary.shp`: favela boundary polygon\n"
        f"- `raw/{slug}_dtm.tif`: terrain raster clipped from `{dtm_filename}` "
        f"(with {terrain_buffer_m:.0f}m buffer)\n"
    )
    readme_path.write_text(readme_text, encoding="utf-8")

    result = {
        "area_name": resolved_name,
        "area_slug": slug,
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
    parser = argparse.ArgumentParser(
        description="Extract data for one Rio favela or complex"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--favela", help="Favela name to look up in the `nome` column (e.g., Rocinha)"
    )
    group.add_argument(
        "--complexo",
        help="Complex name to look up in the `complexo` column (e.g., Complexo do Alemão)",
    )
    parser.add_argument(
        "--dissolve",
        action="store_true",
        help="Dissolve multi-polygon selections into a single boundary geometry",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="Override for the output folder / filename slug (default: derived from name)",
    )
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
    data_rj_dir = Path(args.data_rj_dir)
    output_base_dir = Path(args.output_base_dir)

    boundary_gdf = None
    display_name = None
    area_slug = args.slug

    if args.complexo:
        # --complexo path: look up sub-favelas and optionally dissolve
        favelas_path = data_rj_dir / "Favelas_Limit_2019.shp"
        if not favelas_path.exists():
            raise FileNotFoundError(f"Missing required file: {favelas_path}")
        print("Loading favela boundaries...")
        favelas_gdf = gpd.read_file(favelas_path)
        matches = find_complex(favelas_gdf, args.complexo)
        if matches.empty:
            raise ValueError(
                f"Complex '{args.complexo}' not found in the `complexo` column of {favelas_path.name}"
            )
        name_col = "nome" if "nome" in matches.columns else "NOME"
        print(f"Found {len(matches)} sub-favelas in complex '{args.complexo}':")
        for _, row in matches.iterrows():
            print(f"  - {row[name_col]} (cod_favela={row['cod_favela']})")

        display_name = args.complexo
        if area_slug is None:
            area_slug = normalize_text(args.complexo).replace(" ", "_")

        if args.dissolve:
            dissolved_geom = unary_union(matches.geometry)
            boundary_gdf = gpd.GeoDataFrame(
                {"name": [display_name], "n_components": [len(matches)]},
                geometry=[dissolved_geom],
                crs=matches.crs,
            )
            print(f"Dissolved {len(matches)} sub-favelas into single boundary")
        else:
            boundary_gdf = matches.copy()
            print(
                f"Using {len(matches)} individual sub-favela boundaries (no dissolve)"
            )

        favela_name = args.complexo
    else:
        favela_name = args.favela

    result = extract_favela_data(
        favela_name=favela_name,
        data_rj_dir=data_rj_dir,
        output_base_dir=output_base_dir,
        road_buffer_m=args.road_buffer_m,
        terrain_buffer_m=args.terrain_buffer_m,
        dtm_filename=args.dtm_filename,
        boundary_gdf=boundary_gdf,
        area_slug=area_slug,
        display_name=display_name,
    )
    print("\nExtraction complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
