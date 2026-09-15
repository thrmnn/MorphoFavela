"""WP-04 acceptance: docs/wp04_sites_spec.md, "Tests" section (1-5).

Site data (DTM, footprints, roads, the CPU street-SVF reference) lives only in
the main checkout — data/ and outputs/ are gitignored and were never copied
into this worktree — so those paths are read absolute from the main checkout,
same convention as tests/test_wp02_horizon.py and tests/test_wp05_pilot.py.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import torch
from affine import Affine

from src.brisa_solar import wp02_sky, wp04_sites
from src.brisa_solar.constants import load_params
from src.brisa_solar.wp02_horizon import patch_visibility
from src.svf_v2 import sampling as svf_sampling
from src.svf_v2.compute import generate_tregenza_patches

MAIN_CHECKOUT = Path("/home/theo/SCL/SCR/MorphoFavela")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def directions_weights():
    return generate_tregenza_patches()


@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def sky(params):
    epw = MAIN_CHECKOUT / params["weather"]["primary_epw"]
    if not epw.exists():
        pytest.skip(f"EPW not on disk: {epw}")
    return wp02_sky.build(epw)


@pytest.fixture(scope="module")
def meta(params):
    epw = MAIN_CHECKOUT / params["weather"]["primary_epw"]
    if not epw.exists():
        pytest.skip(f"EPW not on disk: {epw}")
    return wp04_sites.epw_meta(epw)


def _flat_transform(cell: float, size: int):
    origin_x = -size * cell / 2
    origin_y = size * cell / 2
    return Affine(cell, 0, origin_x, 0, -cell, origin_y), origin_x, origin_y


# ---------------------------------------------------------------------------
# 1. Unobstructed flat ground: SVF 1.0, direct-sun hours == pvlib daylight hours
# ---------------------------------------------------------------------------

def test_unobstructed_flat_ground_svf_and_sun_hours(sky, meta, params, directions_weights):
    directions, _weights = directions_weights
    cell, size = 1.0, 40
    transform, _ox, _oy = _flat_transform(cell, size)
    surface = np.zeros((size, size), dtype="float32")
    obs = np.array([[0.0, 0.0]])

    vis, on_building, horizon_deg = patch_visibility(
        surface, transform, obs, directions=directions, device="cpu",
        max_dist_m=wp04_sites.MAX_DIST_M, step_m=cell, return_horizon=True,
    )
    assert not on_building.any()
    svf = sky.svf(vis.astype(float))
    assert svf == pytest.approx(np.ones(1), abs=1e-9)

    patch_az_deg = wp04_sites.patch_azimuth_deg(directions)
    thresholds = params["reference_days"]["duration_thresholds_h"]
    for label in ("winter_solstice", "equinox"):
        date_str = params["reference_days"][label]
        r = wp04_sites.direct_sun_hours(horizon_deg, patch_az_deg, date_str, meta, thresholds)
        assert int(r["hours_count"][0]) == r["daylight_hours_pvlib"], (
            f"{label}: engine direct-sun hours {r['hours_count'][0]} != "
            f"pvlib daylight hours {r['daylight_hours_pvlib']} for unobstructed flat ground")


# ---------------------------------------------------------------------------
# 2. Unobstructed vertical façade: SVF 0.5 +- discretisation tolerance (0.04),
#    for each of four normals.
# ---------------------------------------------------------------------------

def test_unobstructed_vertical_facade_svf(sky, directions_weights):
    directions, weights = directions_weights
    n_patches = len(directions)
    vis = np.ones((4, n_patches), dtype=bool)   # geometrically unobstructed
    normals = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
    ])
    svf, irr = wp04_sites.facade_svf_irradiation(sky, directions, weights, vis, normals)
    assert svf == pytest.approx(np.full(4, 0.5), abs=0.04), (
        f"unobstructed vertical facade SVF {svf} should read ~0.5 (the standard "
        "vertical-surface view factor) for every cardinal normal")
    assert (irr > 0).all() and (irr < float(sky.patch_total_kwh.sum())).all()


# ---------------------------------------------------------------------------
# 3. Single wall north of a point (southern hemisphere): winter-solstice sun is
#    in the NORTHERN sky and is blocked by a wall of height H at distance D iff
#    atan(H/D) > solar altitude.
# ---------------------------------------------------------------------------

def test_single_wall_blocks_winter_solstice_sun_per_pvlib_altitude(meta, params, directions_weights):
    directions, _weights = directions_weights
    D = 20.0
    cell, size = 0.5, 200
    transform, ox, oy = _flat_transform(cell, size)
    ys = oy - (np.arange(size) + 0.5) * cell
    xs = ox + (np.arange(size) + 0.5) * cell
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    thickness = 2 * cell
    # A wall spanning the entire NORTHERN half-plane (y > 0) at radius D — the
    # winter-solstice sun sits north of zenith for Rio (params.yaml
    # hemisphere_note), so this covers whatever azimuth the sun is at near
    # local solar noon without needing to know that azimuth in advance.
    north_wall_mask = lambda H: np.where((R >= D) & (R < D + thickness) & (Y > 0), H, 0.0).astype("float32")

    date_str = params["reference_days"]["winter_solstice"]
    fine = wp04_sites.sun_positions(date_str, meta, "10min")
    peak_i = int(np.argmax(fine["apparent_elevation"].to_numpy()))
    sun_alt = float(fine["apparent_elevation"].to_numpy()[peak_i])
    sun_az = float(fine["azimuth"].to_numpy()[peak_i])
    assert -90.0 < ((sun_az + 180) % 360 - 180) < 90.0, (
        f"premise check: winter-solstice peak-altitude sun azimuth {sun_az} "
        "should be in the northern half (within +-90 deg of due north)")

    patch_az_deg = wp04_sites.patch_azimuth_deg(directions)
    nearest = int(np.argmin(np.abs((sun_az - patch_az_deg + 180.0) % 360.0 - 180.0)))

    obs = np.array([[0.0, 0.0]])
    for H, expect_blocked in [(25.0, True), (10.0, False)]:
        surface = north_wall_mask(H)
        vis, on_building, horizon_deg = patch_visibility(
            surface, transform, obs, directions=directions, device="cpu",
            max_dist_m=100.0, step_m=cell, obs_height_m=0.0, return_horizon=True,
        )
        assert not on_building.any()
        horizon_at_sun = float(horizon_deg[0, nearest])
        engine_visible = sun_alt > horizon_at_sun
        analytic_threshold_deg = np.degrees(np.arctan2(H, D))
        analytic_blocked = analytic_threshold_deg > sun_alt
        assert engine_visible == (not analytic_blocked) == (not expect_blocked), (
            f"H={H}, D={D}: pvlib solar altitude {sun_alt:.2f} deg, "
            f"atan(H/D)={analytic_threshold_deg:.2f} deg, "
            f"engine visible={engine_visible}, analytic blocked={analytic_blocked}")


# ---------------------------------------------------------------------------
# 4. Horizon-angle output equals the binary visibility when compared at the
#    patch altitudes (consistency of the two return values).
# ---------------------------------------------------------------------------

def test_horizon_output_consistent_with_binary_visibility(directions_weights):
    directions, _weights = directions_weights
    d = directions
    cell, size = 1.0, 60
    transform, ox, oy = _flat_transform(cell, size)
    rng = np.random.default_rng(2026091501)
    surface = rng.uniform(0.0, 5.0, size=(size, size)).astype("float32")
    is_building = np.zeros((size, size), dtype=bool)
    is_building[10:15, 10:15] = True
    surface[10:15, 10:15] = 20.0

    obs_rows = rng.integers(0, size, 40)
    obs_cols = rng.integers(0, size, 40)
    xs = ox + (obs_cols + 0.5) * cell
    ys = oy - (obs_rows + 0.5) * cell
    obs = np.stack([xs, ys], axis=1)

    vis, on_building, horizon_deg = patch_visibility(
        surface, transform, obs, directions=directions, is_building=is_building,
        device="cpu", max_dist_m=80, step_m=cell, return_horizon=True,
    )

    alt_deg = np.degrees(np.arcsin(np.clip(d[:, 2], -1.0, 1.0)))
    horiz_norm = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    zenith_patch = horiz_norm <= 1e-9

    reconstructed = (alt_deg[None, :] > horizon_deg.astype("float64")) | zenith_patch[None, :]
    reconstructed = reconstructed & ~on_building[:, None]

    mismatch = np.where(reconstructed != vis)
    n_mismatch = len(mismatch[0])
    # float16 rounding of the stored horizon can flip a comparison only within
    # its own quantisation step of the true (float64) horizon; that step is
    # tiny relative to the ~6-11 deg patch spacing, so any mismatch here must
    # be within that rounding band, never a structural disagreement.
    if n_mismatch:
        rows, cols = mismatch
        band = np.abs(alt_deg[cols] - horizon_deg[rows, cols].astype("float64"))
        assert np.all(band <= 0.05), (
            f"{n_mismatch} mismatches exceed float16 rounding tolerance: max band {band.max()}")
    assert n_mismatch <= int(0.01 * vis.size), f"{n_mismatch}/{vis.size} mismatches, too many for rounding alone"


# ---------------------------------------------------------------------------
# 5. Rio das Pedras street set reproduces the accepted CPU cross-reference.
# ---------------------------------------------------------------------------

def test_riodaspedras_street_crossreference(sky, directions_weights, tmp_path):
    ref_path = MAIN_CHECKOUT / "outputs/riodaspedras/svf_v2/svf_streets.gpkg"
    if not ref_path.exists():
        pytest.skip(f"reference gpkg not on disk: {ref_path}")
    dtm_extended = MAIN_CHECKOUT / "data/riodaspedras/dtm_extended_700m.tif"
    fp_extended = MAIN_CHECKOUT / "data/riodaspedras/buildings_extended_700m.gpkg"
    if not dtm_extended.exists() or not fp_extended.exists():
        pytest.skip(f"site data not on disk: {dtm_extended} / {fp_extended}")

    directions, weights = directions_weights

    surface, transform, _crs, is_building, _bid, _ground, _dtm, _fp = wp04_sites.build_site_surface(
        "riodaspedras", MAIN_CHECKOUT, wp04_sites.CELL_M, tmp_path
    )
    native_dtm, native_fp, native_roads = wp04_sites.resolve_native_paths("riodaspedras", MAIN_CHECKOUT)
    boundary_path = wp04_sites.resolve_native_boundary("riodaspedras", MAIN_CHECKOUT)
    footprints_gdf = gpd.read_file(native_fp)
    boundary_gdf = gpd.read_file(boundary_path) if boundary_path is not None else None
    street_pts = svf_sampling.sample_street_points(
        native_roads, native_dtm, footprints_gdf=footprints_gdf, boundary_gdf=boundary_gdf
    )
    obs_xy = np.column_stack([street_pts.geometry.x.to_numpy(), street_pts.geometry.y.to_numpy()])

    # Raw engine visibility, not evaluate_points' cosine-weighted sky.svf(): the
    # reference gpkg's own "svf" column was produced by the CPU raycaster using a
    # solid-angle/count convention (measured here — unweighted and solid_angle
    # both land at r=0.994, median|delta|=0.014; cosine_weighted, our per-site
    # deliverable's own definition, is a DIFFERENT physical quantity and reads
    # r=0.979 against this same reference by construction, not by engine defect).
    # This test cross-validates the raster horizon-marching GEOMETRY (does it
    # reproduce the CPU raycaster's visibility?), same as test_wp02_sky.py's own
    # crossreference test, which picks whichever variant matches best.
    vis, on_building = patch_visibility(
        surface, transform, obs_xy, directions=directions, is_building=is_building,
        obs_height_m=wp04_sites.OBS_HEIGHT_M, max_dist_m=wp04_sites.MAX_DIST_M,
        march_sampling="nearest", device=DEVICE,
    )

    from src.brisa_solar.wp02_horizon import svf_solid_angle, svf_unweighted
    from scipy.spatial import cKDTree

    ref = gpd.read_file(ref_path)
    ref_xy = np.column_stack([ref.geometry.x.to_numpy(), ref.geometry.y.to_numpy()])
    tree = cKDTree(ref_xy)
    dist, idx = tree.query(obs_xy, k=1)
    matched = (dist < 0.01) & ~on_building
    n_matched = int(matched.sum())
    assert n_matched >= 0.9 * len(ref), (
        f"only {n_matched}/{len(ref)} generated street points matched the reference "
        "within 0.01 m — sample_street_points may have drifted from what produced "
        f"{ref_path}")

    reference = ref["svf"].to_numpy(dtype="float64")[idx[matched]]
    variants = {
        "unweighted": svf_unweighted(vis)[matched],
        "solid_angle": svf_solid_angle(vis, weights)[matched],
    }
    results = {}
    for name, measured in variants.items():
        delta = np.abs(measured - reference)
        results[name] = (float(np.corrcoef(measured, reference)[0, 1]), float(np.median(delta)))
    best_name = max(results, key=lambda n: results[n][0])
    r, median_abs_delta = results[best_name]
    print(f"riodaspedras street crossref: n={n_matched}, best={best_name}, "
          f"r={r:.4f}, median|delta|={median_abs_delta:.4f} ({results})")
    assert r >= 0.98 and median_abs_delta <= 0.03, (
        f"accepted floor failed for best variant '{best_name}': r={r:.4f} (need >=0.98), "
        f"median|delta|={median_abs_delta:.4f} (need <=0.03)")
