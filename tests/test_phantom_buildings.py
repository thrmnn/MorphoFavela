"""Guard against the topo==0 phantom-tower corruption reaching per-patch packages."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

gpd = pytest.importorskip("geopandas")
from run_pilot_sampling import drop_phantom_buildings  # noqa: E402
from shapely.geometry import box  # noqa: E402


def _bld(rows):
    geoms = [box(i, 0, i + 1, 1) for i in range(len(rows))]
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:31983")


def test_drops_topo_zero_and_keeps_healthy():
    g = _bld([
        {"altura": 10.0, "base": 50.0, "topo": 60.0},   # healthy: topo = base+altura
        {"altura": 129.3, "base": 129.3, "topo": 0.0},  # phantom: topo==0, altura==base
        {"altura": 8.0, "base": 12.0, "topo": 20.0},    # healthy
    ])
    out = drop_phantom_buildings(g)
    assert len(out) == 2
    assert (out["topo"] > 0).all()
    assert out["altura"].max() < 40


def test_keeps_real_tall_building():
    # a genuine 45 m block with a consistent topo must NOT be dropped
    g = _bld([{"altura": 45.0, "base": 10.0, "topo": 55.0}])
    assert len(drop_phantom_buildings(g)) == 1


def test_noop_without_height_columns():
    g = _bld([{"foo": 1}, {"foo": 2}])
    assert len(drop_phantom_buildings(g)) == 2
