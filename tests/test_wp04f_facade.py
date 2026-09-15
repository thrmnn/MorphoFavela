"""WP-04F acceptance: docs/wp04f_facade_spec.md, deliverable 5.

Three synthetic checks, no site data required (hermetic, matches the style of
tests/test_wp02_horizon.py's synthetic-raster tests): own-building exclusion
makes a façade point's SVF independent of its inset distance; the exclusion
mechanism reproduces the closed-form opposite-parallel-wall horizon under both
engine variants (with/without own-building exclusion); and build_surface's new
building_id raster round-trips correctly and stays self-consistent with
is_building.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import Polygon

from src.brisa_solar import wp02_sky, wp04_sites
from src.brisa_solar.constants import load_params
from src.brisa_solar.wp02_horizon import patch_visibility
from src.brisa_solar.wp02_surface import build_surface, load_building_id, load_ground, load_surface
from src.svf_v2.compute import generate_tregenza_patches

MAIN_CHECKOUT = Path("/home/theo/SCL/SCR/MorphoFavela")


@pytest.fixture(scope="module")
def directions_weights():
    return generate_tregenza_patches()


@pytest.fixture(scope="module")
def sky():
    epw = MAIN_CHECKOUT / load_params()["weather"]["primary_epw"]
    if not epw.exists():
        pytest.skip(f"EPW not on disk: {epw}")
    return wp02_sky.build(epw)


def _flat_transform(cell: float, size: int):
    origin_x = -size * cell / 2
    origin_y = size * cell / 2
    return Affine(cell, 0, origin_x, 0, -cell, origin_y), origin_x, origin_y


# ---------------------------------------------------------------------------
# 1. Own-building exclusion: a point 0.5 m outside a 10 m cube sees the same
#    sky as a point 1.5 m outside, once the rasterization "leak" that marks a
#    cell just outside the wall as the cube's own roof is excluded.
# ---------------------------------------------------------------------------

def test_own_building_exclusion_matches_across_inset(sky, directions_weights):
    directions, weights = directions_weights
    cell, size = 1.0, 60
    transform, ox, oy = _flat_transform(cell, size)

    surface = np.zeros((size, size), dtype="float64")
    building_id = np.zeros((size, size), dtype=np.int64)
    ground = np.zeros((size, size), dtype="float64")

    # True 10x10 m cube footprint: x in [0, 10), y in [0, 10) -> cols 30-39, rows 20-29.
    cube_rows, cube_cols = slice(20, 30), slice(30, 40)
    surface[cube_rows, cube_cols] = 10.0
    building_id[cube_rows, cube_cols] = 1

    # The rasterization "leak" the spec describes: the cell-centre rule lets the
    # cube's roof spill one cell outward past its own wall face (col 29, x in
    # [-1, 0)) — a genuine part of the defect being fixed, not an artificial one.
    leak_col = 29
    surface[cube_rows, leak_col] = 10.0
    building_id[cube_rows, leak_col] = 1

    obs_xy = np.array([[-0.5, 4.5], [-1.5, 4.5]])  # 0.5 m and 1.5 m outside the wall
    obs_z = np.array([5.0, 5.0])
    normals = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    obs_building = np.array([1, 1])

    # Premise check: without exclusion, the 0.5 m point (nearest cell = the leak
    # cell, col 29) reads heavily self-occluded while the 1.5 m point (col 28,
    # genuinely clear) does not — this is the bug runs/wp04_sites_20260914T230606Z
    # showed as unexplained exact zeros.
    vis_base, _on_b = patch_visibility(
        surface, transform, obs_xy, directions=directions, obs_z=obs_z,
        max_dist_m=25.0, step_m=cell, device="cpu",
    )
    svf_base, _irr_base = wp04_sites.facade_svf_irradiation(sky, directions, weights, vis_base, normals)
    assert svf_base[0] < svf_base[1] - 0.03, (
        f"premise check: baseline (no exclusion) should show the 0.5 m point "
        f"({svf_base[0]:.3f}) measurably more self-occluded than the 1.5 m point "
        f"({svf_base[1]:.3f}) — otherwise this raster doesn't reproduce the bug")

    vis_fix, _on_f = patch_visibility(
        surface, transform, obs_xy, directions=directions, obs_z=obs_z,
        max_dist_m=25.0, step_m=cell, device="cpu",
        obs_building=obs_building, building_id_raster=building_id, ground_surface=ground,
    )
    svf_fix, _irr_fix = wp04_sites.facade_svf_irradiation(sky, directions, weights, vis_fix, normals)

    assert svf_fix == pytest.approx(np.full(2, 0.5), abs=0.04), (
        f"with own-building exclusion, both points should read the standard "
        f"unobstructed vertical-facade SVF ~0.5: got {svf_fix}")
    assert svf_fix[0] == pytest.approx(svf_fix[1], abs=0.01), (
        f"with own-building exclusion, the 0.5 m and 1.5 m points should see "
        f"the same sky: {svf_fix[0]:.4f} vs {svf_fix[1]:.4f}")


# ---------------------------------------------------------------------------
# 2. Opposite-parallel-wall closed form, derived from the same patch-centre
#    rule the WP-02 canyon test uses; must hold under both engine variants
#    (own-building exclusion off and on) since it only ever touches cells
#    matching the observer's own building id, never another building's.
# ---------------------------------------------------------------------------

def test_opposite_wall_closed_form_both_engine_variants(directions_weights):
    directions, _weights = directions_weights
    cell, size = 0.5, 200
    transform, ox, oy = _flat_transform(cell, size)
    ys = oy - (np.arange(size) + 0.5) * cell
    xs = ox + (np.arange(size) + 0.5) * cell
    Y, _X = np.meshgrid(ys, xs, indexing="ij")

    D, H = 20.0, 15.0
    surface = np.where(Y >= D, H, 0.0).astype("float64")
    building_id = np.zeros((size, size), dtype=np.int64)
    ground = np.zeros((size, size), dtype="float64")

    # The observer's OWN building sits entirely behind it (y in [-3, -1]), so it
    # can never intersect a +y-facing ray toward the opposite wall — exclusion
    # should be a complete no-op for those directions, while still being
    # exercised (a -y ray does hit this block).
    own_rows = (Y[:, 0] >= -3.0) & (Y[:, 0] < -1.0)
    own_cols = (xs >= -1.0) & (xs < 1.0)
    surface[np.ix_(own_rows, own_cols)] = 5.0
    building_id[np.ix_(own_rows, own_cols)] = 1

    obs = np.array([[0.0, -0.5]])
    obs_z = np.array([2.0])
    obs_building = np.array([1])
    max_dist_m = 40.0  # < domain half-width (50 m): every included patch's own
                        # wall intersection stays inside the array, no edge clamp

    z_p = float(obs_z[0])
    closed_horizon_deg = wp04_sites.opposite_wall_horizon_deg(directions, H, D, z_p)
    dx, dy = directions[:, 0], directions[:, 1]
    horiz_norm = np.hypot(dx, dy)
    hy = np.divide(dy, horiz_norm, out=np.zeros_like(dy), where=horiz_norm > 1e-9)
    # Front hemisphere for normal (0, 1, 0) AND the ray's own horizontal
    # distance to the wall (D / hy) must land inside the finite march — a
    # grazing patch whose wall intersection lies beyond max_dist_m is excluded
    # by definition (same pattern as test_isolated_wall_shadow's tangent
    # exclusion): the engine correctly reports no obstruction there, the
    # closed form (an infinite-march idealisation) does not know that bound.
    facing_wall = hy >= (D / max_dist_m)

    quantum_deg = np.degrees(np.arctan2(cell, D))  # one march step's own positional quantum, at the wall's distance

    for label, kwargs in [
        ("baseline", {}),
        ("own_building_exclusion", dict(obs_building=obs_building, building_id_raster=building_id, ground_surface=ground)),
    ]:
        vis, on_building, horizon_deg = patch_visibility(
            surface, transform, obs, directions=directions, obs_z=obs_z,
            max_dist_m=max_dist_m, step_m=cell, device="cpu", return_horizon=True, **kwargs,
        )
        engine_deg = horizon_deg[0].astype("float64")
        band = np.abs(engine_deg[facing_wall] - closed_horizon_deg[facing_wall])
        assert np.all(band <= quantum_deg + 0.05), (
            f"{label}: opposite-wall closed form mismatch beyond the march's own "
            f"quantum ({quantum_deg:.3f} deg): max band {band.max():.3f} deg")

    # And the two variants must in fact agree with EACH OTHER on those same
    # wall-facing directions (own-building exclusion is provably a no-op there).
    vis_a, _ob_a, hz_a = patch_visibility(
        surface, transform, obs, directions=directions, obs_z=obs_z,
        max_dist_m=max_dist_m, step_m=cell, device="cpu", return_horizon=True,
    )
    vis_b, _ob_b, hz_b = patch_visibility(
        surface, transform, obs, directions=directions, obs_z=obs_z,
        max_dist_m=max_dist_m, step_m=cell, device="cpu", return_horizon=True,
        obs_building=obs_building, building_id_raster=building_id, ground_surface=ground,
    )
    assert np.allclose(hz_a[0][facing_wall], hz_b[0][facing_wall], atol=1e-3), (
        "own-building exclusion changed the horizon toward a DIFFERENT building's wall")

    # Premise / mechanism check: the exclusion DOES fire for the backward (-y)
    # directions that hit the observer's own block, proving it isn't a silent
    # no-op everywhere (the own block is 5 m tall right behind the observer).
    behind = (dy < -1e-9) & (horiz_norm > 1e-9)
    if behind.any():
        assert np.any(hz_a[0][behind] > hz_b[0][behind] + 1.0), (
            "own-building exclusion should measurably lower the horizon behind "
            "the observer, where its own building sits")


# ---------------------------------------------------------------------------
# 3. building_id raster: written by build_surface, round-trips through
#    load_building_id, and stays consistent with is_building / load_ground.
# ---------------------------------------------------------------------------

def test_building_id_raster_round_trip(tmp_path):
    cell_m = 1.0
    size = 20
    transform = Affine(cell_m, 0, 0.0, 0, -cell_m, float(size))
    dtm = np.full((size, size), 3.0, dtype="float32")

    dtm_path = tmp_path / "dtm.tif"
    with rasterio.open(
        dtm_path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:31983", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(dtm, 1)

    # Two disjoint 4x4 m footprints -> ids 1 and 2 (1-based positional index).
    poly_a = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
    poly_b = Polygon([(10, 10), (14, 10), (14, 14), (10, 14)])
    gdf = gpd.GeoDataFrame(
        {"base": [3.0, 3.0], "altura": [6.0, 9.0], "topo": [np.nan, np.nan]},
        geometry=[poly_a, poly_b], crs="EPSG:31983",
    )
    fp_path = tmp_path / "buildings.gpkg"
    gdf.to_file(fp_path, driver="GPKG")

    out_stem = tmp_path / "test_1m"
    surface_tif = build_surface(dtm_path, fp_path, cell_m, out_stem)
    is_building_tif = surface_tif.with_name(surface_tif.stem.replace("_surface", "_is_building") + ".tif")
    building_id_tif = surface_tif.with_name(surface_tif.stem.replace("_surface", "_building_id") + ".tif")
    ground_tif = surface_tif.with_name(surface_tif.stem.replace("_surface", "_ground") + ".tif")

    surface, _t, _crs, is_building = load_surface(surface_tif, is_building_tif)
    building_id = load_building_id(building_id_tif)
    ground = load_ground(ground_tif)

    assert building_id is not None and building_id.dtype == np.int32
    assert ground is not None
    assert building_id.shape == surface.shape == is_building.shape == ground.shape

    # Consistency: a cell is a building iff its id is nonzero.
    assert np.array_equal(is_building, building_id != 0)

    # Row-position id scheme: poly_a (row 0) -> id 1, poly_b (row 1) -> id 2.
    ids_present = set(np.unique(building_id[building_id != 0]).tolist())
    assert ids_present == {1, 2}, f"expected ids {{1, 2}}, got {ids_present}"

    # poly_a footprint cells carry id 1 and roof height 3 + 6 = 9.
    rows_a, cols_a = np.where(building_id == 1)
    assert len(rows_a) > 0
    assert surface[rows_a, cols_a] == pytest.approx(9.0)

    # poly_b footprint cells carry id 2 and roof height 3 + 9 = 12.
    rows_b, cols_b = np.where(building_id == 2)
    assert len(rows_b) > 0
    assert surface[rows_b, cols_b] == pytest.approx(12.0)

    # ground.tif is the bare DTM (no building tops) on the same grid, everywhere.
    assert ground == pytest.approx(dtm, abs=1e-4)

    # Non-building cells: surface == ground == dtm.
    open_mask = ~is_building
    assert surface[open_mask] == pytest.approx(ground[open_mask])
