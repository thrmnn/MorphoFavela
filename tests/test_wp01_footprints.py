"""The citywide footprint layer mixes CRSs; these pin the detector that repairs it.

`data/RJ/buildings_RJ_2019.shp` declares EPSG:31983 but stores its first
~467,300 features in geographic degrees. The detector is what keeps a fifth of
the city's buildings from silently vanishing out of every projected raster.
"""
from __future__ import annotations

import pytest
from pyproj import Transformer
from shapely.geometry import Polygon

from src.brisa_solar.wp01_footprints import (
    GEOGRAPHIC_EPSG,
    PROJECTED_EPSG,
    looks_geographic,
)


@pytest.mark.parametrize(
    "x, y",
    [
        (680400.0, 7456000.0),   # Vidigal, UTM 23S
        (623533.0, 7446251.2),   # citywide DTM lower-left corner
        (695128.0, 7483371.2),   # citywide DTM upper-right corner
    ],
)
def test_projected_coordinates_are_not_flagged(x, y):
    assert not looks_geographic(x, y)


@pytest.mark.parametrize(
    "lon, lat",
    [
        (-43.2, -22.9),          # central Rio
        (-43.7667, -22.9889),    # west/south extremes actually observed in the layer
        (-43.1623, -22.7843),
    ],
)
def test_geographic_coordinates_are_flagged(lon, lat):
    assert looks_geographic(lon, lat)


def test_reprojected_rio_degrees_land_inside_the_citywide_dtm():
    """The repair must put degree-stored features back over Rio, not somewhere plausible-looking."""
    t = Transformer.from_crs(f"EPSG:{GEOGRAPHIC_EPSG}", f"EPSG:{PROJECTED_EPSG}", always_xy=True)
    x, y = t.transform(-43.2, -22.9)
    # citywide DTM bounds, measured 2026-09-08
    assert 623533.0 <= x <= 695128.0
    assert 7446251.2 <= y <= 7483371.2
    assert not looks_geographic(x, y), "a repaired coordinate must not re-trigger the detector"


def test_detector_is_geometric_not_positional():
    """A polygon's own coordinates decide, so a mixed file is repairable feature-by-feature."""
    geographic = Polygon([(-43.2, -22.9), (-43.2, -22.8999), (-43.1999, -22.8999)])
    projected = Polygon([(680400, 7456000), (680410, 7456000), (680410, 7456010)])
    assert looks_geographic(*geographic.representative_point().coords[0])
    assert not looks_geographic(*projected.representative_point().coords[0])
