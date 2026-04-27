"""Shared fixtures for svf_v2 tests."""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import pyvista as pv
from shapely.geometry import Polygon, box

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AREAS = ["vidigal_tls", "vidigal", "riodaspedras"]


def _area_data_available(area: str) -> bool:
    """Check if real data files exist for an area."""
    from src.config import get_area_data_dir
    from src.svf_v2.paths import AREA_FILES

    if area not in AREA_FILES:
        return False
    data_dir = get_area_data_dir(area)
    return all((data_dir / v).exists() for v in AREA_FILES[area].values())


@pytest.fixture(params=AREAS)
def area_name(request):
    """Parametrized fixture yielding each area name; skips if data missing."""
    if not _area_data_available(request.param):
        pytest.skip(f"Data not available for {request.param}")
    return request.param


@pytest.fixture
def area_paths(area_name):
    """Resolved (dtm_path, footprints_path, roads_path) for the area."""
    from src.svf_v2.paths import resolve_paths

    return resolve_paths(area_name)


@pytest.fixture(scope="session")
def _scene_cache():
    """Session-scoped cache for built scenes (expensive to rebuild)."""
    return {}


@pytest.fixture
def area_scene(area_name, area_paths, _scene_cache):
    """Built scene (scene_mesh, terrain, footprints_gdf) for an area.
    Uses dtm_subsample=4 for speed. Cached per session."""
    if area_name in _scene_cache:
        return _scene_cache[area_name]
    from src.svf_v2.scene import build_scene

    dtm_path, footprints_path, _ = area_paths
    result = build_scene(
        dtm_path,
        footprints_path,
        area=area_name,
        dtm_subsample=4,
    )
    _scene_cache[area_name] = result
    return result


@pytest.fixture
def flat_terrain():
    """10x10 flat terrain mesh at z=0."""
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    grid = pv.StructuredGrid(xx, yy, zz)
    return grid.extract_surface()


def _make_box_building(xmin, ymin, xmax, ymax, base, height):
    """Create an extruded box building mesh."""
    poly = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    coords = np.array(poly.exterior.coords)[:-1]
    n = len(coords)
    xs, ys = coords[:, 0], coords[:, 1]

    bottom = np.column_stack([xs, ys, np.full(n, base)])
    top = np.column_stack([xs, ys, np.full(n, base + height)])
    verts = np.vstack([bottom, top])

    faces = []
    faces.append([n] + list(range(n - 1, -1, -1)))
    faces.append([n] + list(range(n, 2 * n)))
    for i in range(n):
        j = (i + 1) % n
        faces.append([3, i, j, i + n])
        faces.append([3, j, j + n, i + n])

    flat = []
    for f in faces:
        flat.extend(f)
    return pv.PolyData(verts, np.array(flat, dtype=np.int32))


@pytest.fixture
def single_building_scene(flat_terrain):
    """Flat terrain + one 2x2x5 building at centre."""
    bldg = _make_box_building(4, 4, 6, 6, 0, 5)
    return flat_terrain + bldg


@pytest.fixture
def empty_scene(flat_terrain):
    """Flat terrain with no buildings."""
    return flat_terrain


@pytest.fixture
def building_footprints_gdf():
    """GeoDataFrame with one building footprint and base/altura fields."""
    poly = box(4, 4, 6, 6)
    return gpd.GeoDataFrame(
        {"base": [0.0], "altura": [5.0], "geometry": [poly]},
        crs="EPSG:31983",
    )


@pytest.fixture
def sky_directions_small():
    """Small set of sky directions for fast tests."""
    from src.svf_v2.compute import generate_sky_directions

    return generate_sky_directions(n_patches=20)


@pytest.fixture
def sky_directions_default():
    """Default 145-patch sky directions."""
    from src.svf_v2.compute import generate_sky_directions

    return generate_sky_directions(n_patches=145)
