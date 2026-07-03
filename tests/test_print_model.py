"""Print-model geometry: the plinth must be a watertight solid and scaling exact."""

from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.print3d.model import _building_prisms, heightmap_to_solid

ROOT = Path(__file__).resolve().parents[1]


def _flat_grid(n=5, z=10.0):
    xs = np.arange(n) * 5.0
    X, Y = np.meshgrid(xs, xs)
    Z = np.full((n, n), z)
    return X, Y, Z


def test_plinth_is_watertight_solid():
    X, Y, Z = _flat_grid()
    mesh = heightmap_to_solid(X, Y, Z, floor_z=0.0)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0


def test_plinth_volume_matches_box():
    # flat 20×20 m top at z=10 over a floor at z=0 → a 20×20×10 box
    X, Y, Z = _flat_grid(n=5, z=10.0)
    mesh = heightmap_to_solid(X, Y, Z, floor_z=0.0)
    assert mesh.volume == pytest.approx(20 * 20 * 10, rel=1e-6)


def test_relief_preserved_in_solid_bounds():
    X, Y, Z = _flat_grid()
    Z[2, 2] = 40.0  # a peak
    mesh = heightmap_to_solid(X, Y, Z, floor_z=0.0)
    assert mesh.bounds[1][2] == pytest.approx(40.0)
    assert mesh.bounds[0][2] == pytest.approx(0.0)


def test_building_prism_from_3d_footprint():
    """Regression: cadaster footprints carry a Z ordinate. The extruder must
    cope (force_2d + engine-agnostic) — hardcoding the 'triangle' backend used
    to break this whenever that optional package was absent."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    poly3d = Polygon([(0, 0, 5), (4, 0, 5), (4, 3, 5), (0, 3, 5)])  # has_z
    bld = gpd.GeoDataFrame({"base": [0.0], "altura": [9.0]}, geometry=[poly3d])
    prisms = _building_prisms(bld, embed=2.0)
    assert len(prisms) == 1
    assert prisms[0].is_watertight
    assert prisms[0].volume == pytest.approx(4 * 3 * (9.0 + 2.0), rel=1e-6)


# --- full-site DSM heightfield (needs the gitignored data/ tree) --------------

_HAS_VIDIGAL = (ROOT / "data" / "vidigal" / "dtm_extended_300m.tif").exists()
_site = pytest.mark.skipif(not _HAS_VIDIGAL, reason="site data (gitignored) not present")


@_site
def test_site_dsm_grid_is_bounded_and_has_massing():
    """The print grid caps its long side near target/cell regardless of favela
    size, and building rasterisation produces a non-trivial massing layer that
    never sinks the ground."""
    from src.print3d.model import sample_site_dsm

    dsm = sample_site_dsm("vidigal", target_mm=50.0, model_cell_mm=0.25)
    assert max(dsm.height.shape) <= 210  # ~50mm / 0.25mm, plus rounding
    assert (dsm.height > 0).any() and (dsm.height == 0).any()  # both built + open ground
    assert dsm.height.min() == 0.0  # heights are above-ground; never negative
    assert np.isfinite(dsm.surface).all()


@_site
def test_site_model_is_watertight_and_fits_box():
    from src.print3d.model import build_site_model

    mesh, stats, _ = build_site_model("vidigal", target_mm=50.0)
    assert mesh.is_watertight
    assert mesh.volume > 0
    assert max(stats.model_mm[0], stats.model_mm[1]) == pytest.approx(50.0, abs=0.2)


# --- Version A sunlight-texture tile (needs the gitignored patch outputs) ------

_HAS_VDG_PATCH = (
    ROOT / "outputs/vidigal/sampling_cfd/campaign_sampling/patches/VDG-P07/terrain.tif"
).exists()
_tile = pytest.mark.skipif(not _HAS_VDG_PATCH, reason="patch outputs (gitignored) not present")


@_tile
def test_texture_is_ground_only_and_correct_depth():
    """Each treatment displaces ground cells only, never under buildings, and the
    max depth honours the spec (pit 0.5mm, contour step+groove ~1.0mm, hatch 0.3mm)."""
    from src.print3d.texture import VARIANTS, sample_tile

    tile = sample_tile("vidigal", "VDG-P07", tile_mm=150.0, model_cell_mm=1.0)  # coarse=fast
    for name, fn in VARIANTS.items():
        disp = fn(tile)
        assert (disp[~tile.ground_mask] == 0).all(), f"{name} textured a building cell"
        assert disp.min() >= 0
        assert disp.max() > 0
        depth_mm = disp.max() * tile.mm_per_m
        assert depth_mm <= 1.2, f"{name} exceeds spec depth: {depth_mm:.2f} mm"


@_tile
def test_texture_tile_is_watertight():
    from src.print3d.texture import build_tile, sample_tile

    tile = sample_tile("vidigal", "VDG-P07", tile_mm=150.0, model_cell_mm=1.0)
    mesh, stats, _ = build_tile(tile, "stipple")
    assert mesh.is_watertight
    assert mesh.volume > 0
    assert max(stats.model_mm[0], stats.model_mm[1]) == pytest.approx(150.0, abs=1.0)
