"""Shared fixtures for patch_selection tests."""

import numpy as np
import pytest
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box, LineString

CRS = "EPSG:31983"


@pytest.fixture
def synthetic_boundary():
    """400×400m square boundary centred at (680000, 7471000)."""
    poly = box(679800, 7470800, 680200, 7471200)
    return gpd.GeoDataFrame({"geometry": [poly]}, crs=CRS)


@pytest.fixture
def synthetic_buildings():
    """20 buildings scattered in a 400×400m area with altura and base fields."""
    rng = np.random.RandomState(42)
    n = 20
    cx = 679800 + rng.uniform(20, 380, n)
    cy = 7470800 + rng.uniform(20, 380, n)
    sizes = rng.uniform(5, 15, n)
    heights = rng.uniform(3, 20, n)
    bases = rng.uniform(0, 5, n)

    polys = [
        box(x - s / 2, y - s / 2, x + s / 2, y + s / 2)
        for x, y, s in zip(cx, cy, sizes)
    ]
    return gpd.GeoDataFrame(
        {
            "altura": heights,
            "base": bases,
            "geometry": polys,
        },
        crs=CRS,
    )


@pytest.fixture
def synthetic_roads():
    """Grid of road segments in 400×400m area."""
    lines = []
    # Horizontal roads
    for y_off in [100, 200, 300]:
        lines.append(LineString([(679800, 7470800 + y_off), (680200, 7470800 + y_off)]))
    # Vertical roads
    for x_off in [100, 200, 300]:
        lines.append(LineString([(679800 + x_off, 7470800), (679800 + x_off, 7471200)]))
    return gpd.GeoDataFrame({"geometry": lines}, crs=CRS)


@pytest.fixture
def synthetic_dtm(tmp_path):
    """Flat DTM GeoTIFF at 2m resolution covering 400×400m area, elevation=5.0m."""
    path = tmp_path / "test_dtm.tif"
    xmin, ymin, xmax, ymax = 679800, 7470800, 680200, 7471200
    res = 2.0
    cols = int((xmax - xmin) / res)
    rows = int((ymax - ymin) / res)
    data = np.full((rows, cols), 5.0, dtype=np.float32)
    transform = from_bounds(xmin, ymin, xmax, ymax, cols, rows)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def synthetic_tiles(synthetic_boundary):
    """Pre-generated 100m tiles over the synthetic boundary."""
    from src.patch_selection.tiling import generate_tiles, classify_tiles

    tiles = generate_tiles(synthetic_boundary, tile_size=100.0)
    return classify_tiles(tiles, synthetic_boundary)
