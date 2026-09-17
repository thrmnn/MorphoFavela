"""terrain_split acceptance tests.

Real DTM/footprints/EPW/Favelas_Limit_2019/the WP-04 sites run of record live
only in the main checkout (data/ and runs/ are gitignored, never copied into
this worktree) — same convention as tests/test_cityhours.py and
tests/test_wp04_sites.py. These tests exercise the pure array/DataFrame/
raster logic (decomposition math, the flat-reference-surface invariant, the
reproduction check, the fill-fraction blank-raster guard) on synthetic
inputs; anything that needs the real EPW reads it absolute from the main
checkout and skips if it is not on disk.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine

from src.brisa_solar import terrain_split
from src.brisa_solar.constants import load_params
from src.brisa_solar.wp02_surface import build_surface, load_surface

MAIN_CHECKOUT = Path("/home/theo/SCL/SCR/MorphoFavela")


@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def meta(params):
    epw = MAIN_CHECKOUT / params["weather"]["primary_epw"]
    if not epw.exists():
        pytest.skip(f"EPW not on disk: {epw}")
    from src.brisa_solar.wp04_sites import epw_meta

    return epw_meta(epw)


# ---------------------------------------------------------------------------
# (a) open-flat reference hours == pvlib daylight-timestep count
# ---------------------------------------------------------------------------

def test_open_flat_hours_equals_daylight_timestep_count(params, meta):
    from src.brisa_solar.wp04_sites import sun_positions

    date_str = params["reference_days"]["winter_solstice"]
    fine = sun_positions(date_str, meta, "10min")
    expected = float((fine["apparent_elevation"].to_numpy() > 0.0).sum()) / 6.0

    result = terrain_split.open_flat_hours(date_str, meta)
    assert result == pytest.approx(expected, abs=1e-9)
    assert result > 0.0


# ---------------------------------------------------------------------------
# (b) both decomposition orderings sum to the same total loss
# ---------------------------------------------------------------------------

def test_decompose_terrain_first_sums_to_total_loss():
    h_flat = 11.0
    hours_terrain = np.array([9.0, 7.5, 11.0])
    hours_real = np.array([6.0, 5.0, 11.0])

    terrain_loss, buildings_loss = terrain_split.decompose_terrain_first(h_flat, hours_terrain, hours_real)
    total = h_flat - hours_real
    assert terrain_loss == pytest.approx(h_flat - hours_terrain)
    assert buildings_loss == pytest.approx(hours_terrain - hours_real)
    assert (terrain_loss + buildings_loss) == pytest.approx(total)
    assert (terrain_loss >= -1e-9).all()
    assert (buildings_loss >= -1e-9).all()


def test_decompose_buildings_first_sums_to_total_loss():
    h_flat = 11.0
    hours_bldg = np.array([8.5, 6.0, 11.0])
    hours_real = np.array([6.0, 5.0, 11.0])

    buildings_loss, terrain_loss = terrain_split.decompose_buildings_first(h_flat, hours_bldg, hours_real)
    total = h_flat - hours_real
    assert buildings_loss == pytest.approx(h_flat - hours_bldg)
    assert terrain_loss == pytest.approx(hours_bldg - hours_real)
    assert (buildings_loss + terrain_loss) == pytest.approx(total)


def test_the_two_orderings_generally_disagree_on_the_split():
    # Not a tautology: pick numbers where terrain-first and buildings-first
    # give a DIFFERENT split of the SAME total — this is exactly the
    # "attribution is not unique" fact the spec asks to surface, not hide.
    h_flat = 11.0
    hours_terrain = np.array([9.0])
    hours_bldg = np.array([7.0])
    hours_real = np.array([6.0])

    terrain_loss_tf, buildings_loss_tf = terrain_split.decompose_terrain_first(h_flat, hours_terrain, hours_real)
    buildings_loss_bf, terrain_loss_bf = terrain_split.decompose_buildings_first(h_flat, hours_bldg, hours_real)

    total = h_flat - hours_real
    assert (terrain_loss_tf + buildings_loss_tf) == pytest.approx(total)
    assert (terrain_loss_bf + buildings_loss_bf) == pytest.approx(total)
    assert terrain_loss_tf != pytest.approx(terrain_loss_bf)
    assert buildings_loss_tf != pytest.approx(buildings_loss_bf)


# ---------------------------------------------------------------------------
# (c) fill-fraction guard against the "rendered too fine, comes out blank" defect
# ---------------------------------------------------------------------------

def test_fill_fraction_measures_finite_share():
    grid = np.full((10, 10), np.nan, dtype="float32")
    grid[2:5, 2:5] = 1.0  # 9 / 100 finite
    assert terrain_split.fill_fraction(grid) == pytest.approx(0.09)


def test_assert_not_blank_raises_on_a_near_empty_raster():
    grid = np.full((100, 100), np.nan, dtype="float32")
    grid[0, 0] = 1.0  # 1 / 10000 = 0.01%, well under the defect's 0.8-1.4% range
    with pytest.raises(ValueError, match="visually blank"):
        terrain_split.assert_not_blank(grid, "synthetic")


def test_assert_not_blank_passes_on_a_well_filled_raster():
    grid = np.full((10, 10), np.nan, dtype="float32")
    grid[:, :6] = 1.0  # 60% filled
    frac = terrain_split.assert_not_blank(grid, "synthetic")
    assert frac == pytest.approx(0.6)


def test_scatter_to_grid_crops_to_observer_bbox_not_a_wider_raster():
    rows = np.array([100, 100, 102])
    cols = np.array([50, 51, 50])
    values = np.array([1.0, 2.0, 3.0])
    grid, bbox = terrain_split.scatter_to_grid(rows, cols, values)
    assert grid.shape == (3, 2)  # rows 100-102, cols 50-51 — not the full site+halo raster
    assert bbox == (100, 102, 50, 51)
    assert grid[0, 0] == pytest.approx(1.0)
    assert grid[0, 1] == pytest.approx(2.0)
    assert grid[2, 0] == pytest.approx(3.0)
    assert np.isnan(grid[1, 0])


# ---------------------------------------------------------------------------
# (d) reproduction check: identical values pass; a real deviation fails;
# defaults are never loosened (same discipline as cityhours.reproduction_check)
# ---------------------------------------------------------------------------

def test_reproduction_check_passes_on_identical_values(tmp_path):
    from src.brisa_solar.wp07_ledger import RUN_OF_RECORD, SITE_DIRS

    n = 300
    rng = np.random.default_rng(1)
    record = pd.DataFrame({
        "row": np.arange(n), "col": np.arange(n),
        "hours_winter_solstice": rng.uniform(0, 11, size=n),
    })
    site_dir = tmp_path / "runs" / RUN_OF_RECORD["wp04"] / SITE_DIRS["vidigal"]
    site_dir.mkdir(parents=True)
    record.to_parquet(site_dir / "ground.parquet", index=False)

    cells = pd.DataFrame({
        "row": record["row"], "col": record["col"],
        "hours_real_winter_solstice": record["hours_winter_solstice"],
    })
    result = terrain_split.reproduction_check({"vidigal": cells}, tmp_path, ["winter_solstice"])

    assert result["pass"] is True
    assert result["per_site"]["vidigal"]["winter_solstice"]["corr"] == pytest.approx(1.0, abs=1e-9)
    assert result["per_site"]["vidigal"]["winter_solstice"]["max_abs_diff"] == pytest.approx(0.0, abs=1e-12)


def test_reproduction_check_fails_on_a_real_deviation(tmp_path):
    from src.brisa_solar.wp07_ledger import RUN_OF_RECORD, SITE_DIRS

    n = 300
    rng = np.random.default_rng(2)
    record = pd.DataFrame({
        "row": np.arange(n), "col": np.arange(n),
        "hours_winter_solstice": rng.uniform(0, 11, size=n),
    })
    site_dir = tmp_path / "runs" / RUN_OF_RECORD["wp04"] / SITE_DIRS["vidigal"]
    site_dir.mkdir(parents=True)
    record.to_parquet(site_dir / "ground.parquet", index=False)

    cells = pd.DataFrame({
        "row": record["row"], "col": record["col"],
        "hours_real_winter_solstice": record["hours_winter_solstice"] + 0.5,  # past max_abs_diff_max=0.01
    })
    result = terrain_split.reproduction_check({"vidigal": cells}, tmp_path, ["winter_solstice"])

    assert result["pass"] is False
    assert result["per_site"]["vidigal"]["winter_solstice"]["pass"] is False


def test_reproduction_check_never_loosens_tolerance_by_default():
    import inspect

    sig = inspect.signature(terrain_split.reproduction_check)
    assert sig.parameters["corr_min"].default == 0.9999
    assert sig.parameters["max_abs_diff_max"].default == 0.01


# ---------------------------------------------------------------------------
# (e) the flat-reference-surface invariant the buildings-first sensitivity
# split depends on: building TOP heights don't move when the input DTM does
# (wp02_surface.build_surface reads them from footprint attributes only) —
# only the ground level around/under them changes.
# ---------------------------------------------------------------------------

def _write_square_footprint(path: Path, base: float, altura: float, crs="EPSG:31983"):
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame(
        {"base": [base], "altura": [altura], "topo": [float("nan")]},
        geometry=[Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])],
        crs=crs,
    )
    gdf.to_file(path, driver="GPKG")
    return path


def test_building_top_is_invariant_to_the_input_dtm_elevation(tmp_path):
    params = load_params()
    fp_params = params["footprints"]
    assert fp_params["base_attr"] == "base" and fp_params["height_attr"] == "altura"

    fp_path = _write_square_footprint(tmp_path / "fp.gpkg", base=10.0, altura=3.0)

    transform = Affine(1.0, 0, 0, 0, -1.0, 10.0)
    shape = (10, 10)
    profile = dict(driver="GTiff", height=shape[0], width=shape[1], count=1,
                   dtype="float32", crs="EPSG:31983", transform=transform, nodata=None)

    # Both constants must stay BELOW the building's absolute top (base + altura
    # = 13.0): build_surface takes max(dtm, building_top) per cell, so a DTM
    # elevation above the building's top would swallow it entirely (surface ==
    # dtm, no building visible) — that is a real, disclosed limitation of the
    # flat-reference approach on steep terrain (see run_site's
    # flat_reference_elevation_method note), not something this invariant test
    # is checking.
    dtm_a = tmp_path / "dtm_a.tif"
    with rasterio.open(dtm_a, "w", **profile) as dst:
        dst.write(np.full(shape, 5.0, dtype="float32"), 1)
    dtm_b = tmp_path / "dtm_b.tif"
    with rasterio.open(dtm_b, "w", **profile) as dst:
        dst.write(np.full(shape, 9.0, dtype="float32"), 1)

    surf_a_tif = build_surface(dtm_a, fp_path, 1.0, tmp_path / "a")
    surf_b_tif = build_surface(dtm_b, fp_path, 1.0, tmp_path / "b")
    ib_a_tif = surf_a_tif.with_name(surf_a_tif.stem.replace("_surface", "_is_building") + ".tif")
    ib_b_tif = surf_b_tif.with_name(surf_b_tif.stem.replace("_surface", "_is_building") + ".tif")

    surf_a, _t_a, _c_a, is_building_a = load_surface(surf_a_tif, ib_a_tif)
    surf_b, _t_b, _c_b, is_building_b = load_surface(surf_b_tif, ib_b_tif)

    assert is_building_a.any(), "the synthetic footprint must rasterise onto at least one cell"
    np.testing.assert_array_equal(is_building_a, is_building_b)
    # building TOP (the surface value on a building cell) is the SAME absolute
    # elevation (base + altura = 13.0) regardless of which DTM was passed in —
    # only the ground elsewhere (10.0 vs 100.0) differs.
    np.testing.assert_allclose(surf_a[is_building_a], surf_b[is_building_b])
    np.testing.assert_allclose(surf_a[is_building_a], 13.0, atol=1e-4)
    np.testing.assert_allclose(surf_a[~is_building_a], 5.0, atol=1e-4)
    np.testing.assert_allclose(surf_b[~is_building_b], 9.0, atol=1e-4)


def test_write_flat_footprints_discards_absolute_elevation_keeps_real_height(tmp_path):
    # The bug this guards against (measured 2026-09-17, Vidigal pilot): using
    # the REAL base/topo on a flat DTM reproduces each building's absolute
    # site elevation almost undisturbed on steep terrain, so a hillside
    # building towers unrealistically over the flat reference (mean
    # buildings_first terrain_loss went negative). write_flat_footprints
    # overrides base to a fixed elevation and drops topo, so only altura
    # (real physical height) survives — two buildings at very different real
    # site elevations but the SAME height must come out with the SAME top.
    from shapely.geometry import Polygon

    params = load_params()
    fp = params["footprints"]
    gdf = gpd.GeoDataFrame(
        {fp["base_attr"]: [10.0, 200.0], fp["height_attr"]: [5.0, 5.0], fp["top_attr"]: [float("nan"), float("nan")]},
        geometry=[Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]), Polygon([(6, 6), (8, 6), (8, 8), (6, 8)])],
        crs="EPSG:31983",
    )
    fp_path = tmp_path / "fp.gpkg"
    gdf.to_file(fp_path, driver="GPKG")

    flat_fp_path = terrain_split.write_flat_footprints(fp_path, 0.0, tmp_path / "flat_fp.gpkg", fp)

    transform = Affine(1.0, 0, 0, 0, -1.0, 10.0)
    shape = (10, 10)
    profile = dict(driver="GTiff", height=shape[0], width=shape[1], count=1,
                   dtype="float32", crs="EPSG:31983", transform=transform, nodata=None)
    flat_dtm = tmp_path / "flat_dtm.tif"
    with rasterio.open(flat_dtm, "w", **profile) as dst:
        dst.write(np.zeros(shape, dtype="float32"), 1)

    surf_tif = build_surface(flat_dtm, flat_fp_path, 1.0, tmp_path / "bldg_only")
    ib_tif = surf_tif.with_name(surf_tif.stem.replace("_surface", "_is_building") + ".tif")
    surf, _t, _c, is_building = load_surface(surf_tif, ib_tif)

    assert is_building.any()
    np.testing.assert_allclose(surf[is_building], 5.0, atol=1e-4)  # both buildings: 0 + altura(5.0)
    np.testing.assert_allclose(surf[~is_building], 0.0, atol=1e-4)


def test_build_flat_reference_dtm_writes_a_constant_grid(tmp_path):
    ground = np.random.default_rng(0).uniform(0, 100, size=(5, 5)).astype("float32")
    transform = Affine(1.0, 0, 0, 0, -1.0, 5.0)
    out = terrain_split.build_flat_reference_dtm(ground, transform, "EPSG:31983", 42.5, tmp_path / "flat.tif")

    with rasterio.open(out) as src:
        arr = src.read(1)
        assert src.transform == transform
    np.testing.assert_allclose(arr, 42.5)


# ---------------------------------------------------------------------------
# (f) never a typed site list / patch count — the module imports both
# ---------------------------------------------------------------------------

def test_site_order_comes_from_the_ledger_not_a_typed_list():
    from src.brisa_solar.wp07_ledger import FAVELAS, SITES

    assert list(terrain_split.SITES) == list(SITES)
    assert set(terrain_split.FAVELAS) == set(FAVELAS)


def test_attribution_choice_is_terrain_first_and_documented():
    assert terrain_split.ATTRIBUTION_CHOICE == "terrain_first"
