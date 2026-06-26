"""
3D scene construction from DTM and building footprints.

Everything stays in world coordinates (EPSG:31983 / SIRGAS 2000 UTM 23S).
No local-coordinate STL translation.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyvista as pv
import rasterio
from rasterio.transform import xy
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_dtm(dtm_path: Path) -> tuple[np.ndarray, rasterio.Affine, object, tuple]:
    """Load DTM raster and return (array, transform, crs, bounds)."""
    logger.info(f"Loading DTM from {dtm_path}")
    with rasterio.open(dtm_path) as src:
        dtm_array = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    valid_mask = np.isfinite(dtm_array) & (dtm_array < 10000)
    dtm_array = np.where(valid_mask, dtm_array, np.nan)

    valid = dtm_array[np.isfinite(dtm_array)]
    if len(valid) > 0:
        logger.info(
            f"  Shape {dtm_array.shape}, elevation {np.nanmin(valid):.1f}-{np.nanmax(valid):.1f}m"
        )
    else:
        logger.warning("  No valid elevation values in DTM!")

    return dtm_array, transform, crs, bounds


def sample_dtm_at_point(
    dtm_array: np.ndarray,
    transform: rasterio.Affine,
    x: float,
    y: float,
) -> float:
    """Sample DTM elevation at a point (nearest-neighbor)."""
    row, col = rasterio.transform.rowcol(transform, x, y)
    if row < 0 or row >= dtm_array.shape[0] or col < 0 or col >= dtm_array.shape[1]:
        return np.nan
    row = int(np.clip(row, 0, dtm_array.shape[0] - 1))
    col = int(np.clip(col, 0, dtm_array.shape[1] - 1))
    return float(dtm_array[row, col])


def sample_dtm_at_points(
    dtm_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Batch-sample DTM elevations using rasterio.sample (fast)."""
    coords = list(zip(xs, ys))
    with rasterio.open(dtm_path) as src:
        vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float64)
    # Clean invalid values
    vals[~np.isfinite(vals)] = np.nan
    vals[vals > 10000] = np.nan
    return vals


def build_terrain_mesh(
    dtm_path: Path,
    bbox: tuple[float, float, float, float] | None = None,
    subsample: int = 1,
) -> pv.PolyData:
    """
    Build a terrain surface mesh from a DTM raster.

    Args:
        dtm_path: Path to DTM GeoTIFF.
        bbox: Optional (minx, miny, maxx, maxy) crop in world coords.
        subsample: Take every Nth pixel (1 = full resolution).

    Returns:
        pv.PolyData terrain mesh in world CRS.
    """
    dtm_array, transform, crs, bounds = load_dtm(dtm_path)

    if subsample > 1:
        dtm_array = dtm_array[::subsample, ::subsample]
        transform = rasterio.Affine(
            transform.a * subsample,
            transform.b,
            transform.c,
            transform.d,
            transform.e * subsample,
            transform.f,
        )

    height, width = dtm_array.shape

    # Build world-coordinate arrays using vectorised xy()
    rows_1d = np.arange(height)
    cols_1d = np.arange(width)
    cols_grid, rows_grid = np.meshgrid(cols_1d, rows_1d)
    x_coords, y_coords = xy(transform, rows_grid.ravel(), cols_grid.ravel())
    x_coords = np.array(x_coords).reshape(height, width)
    y_coords = np.array(y_coords).reshape(height, width)

    # Optional bbox crop
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        mask_x = (x_coords[0, :] >= minx) & (x_coords[0, :] <= maxx)
        mask_y = (y_coords[:, 0] >= miny) & (y_coords[:, 0] <= maxy)
        x_coords = x_coords[np.ix_(mask_y, mask_x)]
        y_coords = y_coords[np.ix_(mask_y, mask_x)]
        dtm_array = dtm_array[np.ix_(mask_y, mask_x)]

    # Fill NaN with minimum valid elevation
    valid = dtm_array[np.isfinite(dtm_array)]
    default_z = float(np.nanmin(valid)) if len(valid) > 0 else 0.0
    z_coords = np.where(np.isfinite(dtm_array), dtm_array, default_z)

    grid = pv.StructuredGrid(x_coords, y_coords, z_coords)
    terrain_mesh = grid.extract_surface()

    logger.info(f"  Terrain mesh: {terrain_mesh.n_points} pts, {terrain_mesh.n_cells} cells")
    return terrain_mesh


def _extrude_polygon(
    footprint: Polygon,
    base_elev: float,
    height: float,
) -> pv.PolyData | None:
    """Extrude a single polygon into a closed 3D solid."""
    if height <= 0 or np.isnan(height) or np.isnan(base_elev):
        return None

    coords = np.array(footprint.exterior.coords)
    if len(coords) < 4:  # need 3 unique + closing duplicate
        return None

    xs = coords[:-1, 0]
    ys = coords[:-1, 1]
    n = len(xs)
    if n < 3:
        return None

    bottom = np.column_stack([xs, ys, np.full(n, base_elev)])
    top = np.column_stack([xs, ys, np.full(n, base_elev + height)])
    verts = np.vstack([bottom, top])

    faces = []
    # Bottom face (reversed winding for downward normal)
    faces.append([n] + list(range(n - 1, -1, -1)))
    # Top face
    faces.append([n] + list(range(n, 2 * n)))
    # Side triangles
    for i in range(n):
        j = (i + 1) % n
        faces.append([3, i, j, i + n])
        faces.append([3, j, j + n, i + n])

    flat = []
    for f in faces:
        flat.extend(f)

    return pv.PolyData(verts, np.array(flat, dtype=np.int32))


def build_building_meshes(
    footprints_path: Path,
    dtm_path: Path,
    height_field: str = "altura",
    base_field: str = "base",
    use_dtm_for_base: bool = False,
    area: str | None = None,
) -> tuple[pv.PolyData | None, gpd.GeoDataFrame]:
    """
    Build extruded building meshes from footprints.

    Uses the ``base`` field for base elevation and ``altura`` for extrusion
    height.  Falls back to DTM sampling when the base field is NaN or when
    ``use_dtm_for_base=True``.

    Args:
        footprints_path: Path to building footprints (shapefile / gpkg).
        dtm_path: Path to DTM raster (for CRS matching and fallback base elev).
        height_field: Column with building extrusion height (default ``"altura"``).
        base_field: Column with building base elevation (default ``"base"``).
        use_dtm_for_base: Always sample base elevation from DTM.
        area: Area name for optional filtering (informal vs formal).

    Returns:
        (combined_mesh, filtered_gdf) -- mesh may be None if no valid buildings.
    """
    from src.svf_v2.utils import filter_buildings

    gdf = gpd.read_file(footprints_path)
    logger.info(f"Loaded {len(gdf)} building footprints (CRS {gdf.crs})")

    # Reproject to DTM CRS if needed
    with rasterio.open(dtm_path) as src:
        dtm_crs = src.crs
    if gdf.crs is not None and gdf.crs != dtm_crs:
        logger.info(f"  Reprojecting footprints from {gdf.crs} to {dtm_crs}")
        gdf = gdf.to_crs(dtm_crs)

    # Validate required fields
    if height_field not in gdf.columns:
        raise ValueError(f"Height field '{height_field}' not in columns: {list(gdf.columns)}")
    if base_field not in gdf.columns and not use_dtm_for_base:
        logger.warning(f"Base field '{base_field}' missing -- will sample from DTM")
        use_dtm_for_base = True

    # Drop registry rows whose absolute top elevation `topo` is 0 — an
    # impossible value on elevated/coastal terrain. In the Rio edificações
    # registry these rows carry a corrupt `altura` mis-derived as the base
    # elevation (Rocinha 33, Rio das Pedras 8, Maré 22), which would extrude
    # into phantom towers up to 232 m. True height is unrecoverable, so drop.
    if "topo" in gdf.columns:
        corrupt = gdf["topo"] == 0
        if corrupt.any():
            logger.warning(
                f"  Dropping {int(corrupt.sum())} building(s) with corrupt height (topo == 0)"
            )
            gdf = gdf[~corrupt].copy()

    # Optional filtering for informal areas
    # skip_area_filter=True: large buildings (schools, commercial) still
    # physically obstruct the sky and must be present in the SVF scene.
    gdf, _, _ = filter_buildings(gdf, area=area, skip_area_filter=True)
    logger.info(f"  After filtering: {len(gdf)} buildings")

    dtm_array, transform, _, _ = load_dtm(dtm_path)

    meshes = []
    skipped = 0

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Extruding buildings"):
        geom = row.geometry
        if geom is None or geom.is_empty:
            skipped += 1
            continue

        # Collect polygons (handle MultiPolygon)
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

        for poly in polys:
            if not isinstance(poly, Polygon):
                skipped += 1
                continue

            # --- Base elevation ---
            if use_dtm_for_base:
                base_elev = _sample_base_from_dtm(poly, dtm_array, transform)
            else:
                base_elev = row[base_field]
                if pd.isna(base_elev):
                    base_elev = _sample_base_from_dtm(poly, dtm_array, transform)

            # --- Height ---
            h = row[height_field]
            if pd.isna(h) or h <= 0:
                skipped += 1
                continue

            if np.isnan(base_elev):
                skipped += 1
                continue

            mesh = _extrude_polygon(poly, float(base_elev), float(h))
            if mesh is not None:
                meshes.append(mesh)

    logger.info(f"  Extruded {len(meshes)} building solids (skipped {skipped})")

    if not meshes:
        return None, gdf

    combined = pv.merge(meshes)

    return combined, gdf


def _sample_base_from_dtm(
    poly: Polygon,
    dtm_array: np.ndarray,
    transform: rasterio.Affine,
) -> float:
    """Sample base elevation from DTM using median of centroid + vertices."""
    samples = []
    cx, cy = poly.centroid.x, poly.centroid.y
    samples.append(sample_dtm_at_point(dtm_array, transform, cx, cy))

    coords = np.array(poly.exterior.coords)
    for x, y in coords[:-1]:
        samples.append(sample_dtm_at_point(dtm_array, transform, x, y))

    valid = [s for s in samples if np.isfinite(s)]
    if not valid:
        return np.nan
    return float(np.median(valid))


def build_scene(
    dtm_path: Path,
    footprints_path: Path,
    height_field: str = "altura",
    base_field: str = "base",
    use_dtm_for_base: bool = False,
    area: str | None = None,
    dtm_subsample: int = 1,
    cache_vtk: Path | None = None,
) -> tuple[pv.PolyData, pv.PolyData, gpd.GeoDataFrame]:
    """
    Build a complete 3D scene (terrain + buildings) in world coordinates.

    Args:
        dtm_path: Path to DTM GeoTIFF.
        footprints_path: Path to building footprints.
        height_field: Column with extrusion height (default ``"altura"``).
        base_field: Column with base elevation (default ``"base"``).
        use_dtm_for_base: Force DTM sampling for base elevation.
        area: Area name for filtering.
        dtm_subsample: DTM subsampling factor.
        cache_vtk: If set, save/load combined scene as VTK file.

    Returns:
        (full_scene, terrain_mesh, footprints_gdf)
    """
    # Derive sibling cache paths for terrain and footprints
    if cache_vtk:
        _terrain_vtk = cache_vtk.parent / f"{cache_vtk.stem}_terrain.vtk"
        _footprints_gpkg = cache_vtk.parent / f"{cache_vtk.stem}_footprints.gpkg"
    else:
        _terrain_vtk = None
        _footprints_gpkg = None

    if cache_vtk and cache_vtk.exists() and _terrain_vtk.exists() and _footprints_gpkg.exists():
        logger.info(f"Loading cached scene from {cache_vtk}")
        full_scene = pv.read(str(cache_vtk))
        terrain = pv.read(str(_terrain_vtk))
        gdf = gpd.read_file(str(_footprints_gpkg))
        return full_scene, terrain, gdf

    terrain = build_terrain_mesh(dtm_path, subsample=dtm_subsample)
    buildings, gdf = build_building_meshes(
        footprints_path,
        dtm_path,
        height_field=height_field,
        base_field=base_field,
        use_dtm_for_base=use_dtm_for_base,
        area=area,
    )

    if buildings is not None:
        full_scene = terrain + buildings
    else:
        full_scene = terrain

    logger.info(f"Full scene: {full_scene.n_points} pts, {full_scene.n_cells} cells")

    if cache_vtk:
        cache_vtk.parent.mkdir(parents=True, exist_ok=True)
        full_scene.save(str(cache_vtk))
        terrain.save(str(_terrain_vtk))
        gdf.to_file(str(_footprints_gpkg), driver="GPKG")
        logger.info(f"  Cached scene to {cache_vtk}")

    return full_scene, terrain, gdf
