"""Shared fixtures for tests/test_solar/ test suite."""

import pytest
import pyvista as pv
import geopandas as gpd
from shapely.geometry import Point


@pytest.fixture
def flat_ground_mesh():
    """10x10m flat ground at z=0."""
    grid = pv.Plane(
        center=(5, 5, 0), direction=(0, 0, 1),
        i_size=10, j_size=10, i_resolution=4, j_resolution=4,
    )
    return grid


@pytest.fixture
def enclosed_mesh():
    """Box enclosure: floor + 4 walls + ceiling around (5,5). No sun can reach interior."""
    box = pv.Box(bounds=(0, 10, 0, 10, 0, 10))
    return box


@pytest.fixture
def half_sky_mesh():
    """Ground plane + single wall on the south side blocking sun from the south."""
    ground = pv.Plane(
        center=(5, 5, 0), direction=(0, 0, 1),
        i_size=20, j_size=20, i_resolution=4, j_resolution=4,
    )
    wall = pv.Plane(
        center=(5, 0, 5), direction=(0, 1, 0),
        i_size=20, j_size=10, i_resolution=4, j_resolution=4,
    )
    return ground + wall


@pytest.fixture
def sample_street_gdf():
    """Simple street GeoDataFrame with z_observer column."""
    return gpd.GeoDataFrame(
        {
            "z_observer": [1.5, 1.5, 1.5],
            "street_id": [0, 0, 1],
            "svf": [0.8, 0.5, 1.0],
        },
        geometry=[Point(5, 5), Point(3, 3), Point(8, 8)],
        crs="EPSG:31983",
    )
