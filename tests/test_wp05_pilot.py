"""WP-05 pilot acceptance: spec docs/wp05_pilot_spec.md, "Tests" section (1-5).

Data (DTM, footprints, EPW) lives only in the main checkout — data/ and
runs/ are gitignored and were never copied into this worktree — so the real
pilot run happens outside pytest; these tests exercise the sampling frame,
stratified draw, tile/halo windowing, visibility packing and manifest
plumbing on synthetic inputs plus the (optionally skipped) shared EPW.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from affine import Affine

from pathlib import Path

from src.brisa_solar import wp02_sky, wp05_pilot
from src.brisa_solar.constants import P1_SKY_PATCHES, load_params
from src.brisa_solar.wp02_horizon import patch_visibility, write_run_manifest
from src.svf_v2.compute import generate_tregenza_patches

# Real DTM/footprints/EPW live only in the main checkout (data/ is gitignored
# and was never copied into this worktree) — same convention as
# tests/test_wp02_horizon.py.
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


# ---------------------------------------------------------------------------
# 1. Frame rules applied in order, each removal recorded
# ---------------------------------------------------------------------------

def test_frame_rules_applied_in_order_with_recorded_removals():
    size = 50
    cell_m = 5.0
    rng = np.random.default_rng(0)
    dtm = rng.uniform(0.0, 10.0, size=(size, size)).astype("float32")
    dtm[:5, :] = np.nan  # "outside the municipality" strip

    is_building = np.zeros((size, size), dtype=bool)
    is_building[20:26, 20:26] = True  # 36 building cells, known exactly

    frame = wp05_pilot.build_frame(
        dtm, is_building, cell_m=cell_m,
        fabric_coverage_threshold=0.05, fabric_footprint_distance_m=15.0,
    )

    rc = frame["removal_counts"]
    assert rc["rule_order"] == [
        "not_building_cell",
        "within_municipality_dtm_valid",
        "fabric_coverage_ge_threshold",
        "within_fabric_footprint_distance_m",
    ]
    # rule 1 is exact and independent of the other rules by construction
    assert rc["removed_building_cells"] == int(is_building.sum()) == 36

    total = frame["total_cells"]
    frame_cells = frame["frame_cells"]
    # sequential application: each rule's removal is counted against the
    # survivors of the previous rule, so the four counts plus what's left
    # must exactly partition the grid — this is what "applied in order" means.
    assert (
        rc["removed_building_cells"]
        + rc["removed_invalid_dtm_cells"]
        + rc["removed_low_fabric_coverage_cells"]
        + rc["removed_far_from_footprint_cells"]
        + frame_cells
    ) == total

    in_frame = frame["in_frame"]
    assert not in_frame[is_building].any()
    assert not np.isnan(dtm[in_frame]).any()
    assert (frame["coverage"][in_frame] >= 0.05 - 1e-9).all()
    assert (frame["dist_m"][in_frame] <= 15.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# 2. Stratified draw: proportional, floor respected, seed reproducible
# ---------------------------------------------------------------------------

def test_stratified_draw_proportional_floor_and_seed_reproducible():
    size = 200
    frame_mask = np.zeros((size, size), dtype=bool)
    stratum = np.zeros((size, size), dtype="int16")

    frame_mask[:50, :100] = True   # 5000 cells -> stratum 0 (well above the floor)
    stratum[:50, :100] = 0
    frame_mask[100, :10] = True    # 10 cells -> stratum 1 (below the floor)
    stratum[100, :10] = 1

    rows1, cols1, strata1, per1 = wp05_pilot.stratified_pilot(
        frame_mask, stratum, pilot_fraction=0.01, floor=50, seed=42,
    )
    assert per1[0]["frame_n"] == 5000
    assert per1[0]["pilot_n"] == 50          # round(0.01 * 5000) == 50, above the floor
    assert per1[1]["frame_n"] == 10
    assert per1[1]["pilot_n"] == 10          # floor(50) capped at the stratum's own 10 cells
    assert len(rows1) == 50 + 10
    assert (strata1 == 0).sum() == 50
    assert (strata1 == 1).sum() == 10

    rows2, cols2, strata2, _per2 = wp05_pilot.stratified_pilot(
        frame_mask, stratum, pilot_fraction=0.01, floor=50, seed=42,
    )
    assert np.array_equal(rows1, rows2) and np.array_equal(cols1, cols2)
    assert np.array_equal(strata1, strata2)

    rows3, cols3, _s3, _p3 = wp05_pilot.stratified_pilot(
        frame_mask, stratum, pilot_fraction=0.01, floor=50, seed=99,
    )
    assert not (np.array_equal(rows1, rows3) and np.array_equal(cols1, cols3))


# ---------------------------------------------------------------------------
# 3. Tile/halo assembly matches the un-tiled surface for an edge observer
# ---------------------------------------------------------------------------

def test_tile_halo_assembly_matches_untiled_surface(directions_weights):
    directions, _weights = directions_weights
    cell = 2.0
    size = 1000  # world extent 0..2000 m on a side
    transform = Affine(cell, 0, 0.0, 0, -cell, size * cell)
    cols = (np.arange(size) + 0.5) * cell
    X = np.broadcast_to(cols, (size, size)).astype("float32")

    wall_lo, wall_hi, H = 450.0, 460.0, 20.0  # a wall just outside a 400 m tile, inside its halo
    surface = np.where((X >= wall_lo) & (X < wall_hi), H, 0.0).astype("float32")

    tile_bounds_ = (0.0, 0.0, 400.0, 2000.0)
    halo_m = 500.0
    hb = wp05_pilot.halo_bounds(tile_bounds_, halo_m=halo_m)

    obs = np.array([[395.0, 1000.0]])  # 5 m inside the tile, near the x=400 edge

    vis_full, _ = patch_visibility(
        surface, transform, obs, directions=directions, device="cpu",
        max_dist_m=halo_m, step_m=cell, obs_height_m=0.0,
    )
    assert not vis_full.all(), "test is vacuous unless the wall actually occludes something"

    cropped, cropped_transform, _ib = wp05_pilot.crop_surface_to_bounds(surface, transform, hb)
    assert cropped.shape[1] < surface.shape[1], "the halo crop must remove columns for this to test anything"

    vis_tiled, _ = patch_visibility(
        cropped, cropped_transform, obs, directions=directions, device="cpu",
        max_dist_m=halo_m, step_m=cell, obs_height_m=0.0,
    )

    assert np.array_equal(vis_full, vis_tiled)


# ---------------------------------------------------------------------------
# 4. Packed visibility round-trips and feeds CumulativeSky.svf unchanged
# ---------------------------------------------------------------------------

def test_packed_visibility_roundtrips_and_feeds_svf_unchanged(sky, directions_weights):
    directions, _weights = directions_weights
    rng = np.random.default_rng(7)
    n = 25
    vis = rng.random((n, len(directions))) < 0.4

    packed = wp05_pilot.pack_visibility(vis)
    unpacked = wp05_pilot.unpack_visibility(packed, len(directions))
    assert np.array_equal(vis, unpacked)

    svf_before = sky.svf(vis.astype(float))
    svf_after = sky.svf(unpacked.astype(float))
    assert np.array_equal(svf_before, svf_after)

    irr_before = sky.irradiation(vis.astype(float))
    irr_after = sky.irradiation(unpacked.astype(float))
    assert np.array_equal(irr_before, irr_after)


# ---------------------------------------------------------------------------
# 5. Manifest carries sky.patches; the citywide consistency test still passes
# ---------------------------------------------------------------------------

def test_manifest_carries_sky_patches(tmp_path):
    manifest_path = write_run_manifest(
        tmp_path, cell_m=5.0, obs_height_m=1.5, max_dist_m=500.0, step_m=5.0,
        sampling_rule="wp05 pilot test manifest", device="cpu",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sky"]["patches"] == P1_SKY_PATCHES
