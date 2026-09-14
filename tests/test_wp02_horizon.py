"""WP-02 acceptance: engine-independent checks first, cross-reference last.

Spec: docs/wp02_horizon_engine_spec.md, "Acceptance" section (tests 1-6).
Large data (DTM, footprints, the CPU-raycaster reference gpkg) lives only in
the main checkout, not this worktree — data/ and outputs/ are gitignored and
were never copied into the worktree — so those paths are read absolute from
the main checkout rather than via REPO_ROOT.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
from affine import Affine

from src.brisa_solar import wp02_sky
from src.brisa_solar.constants import load_params
from src.brisa_solar.wp02_horizon import (
    patch_visibility,
    svf_solid_angle,
    svf_unweighted,
    write_run_manifest,
)
from src.brisa_solar.wp02_surface import build_surface, load_surface
from src.svf_v2.compute import generate_tregenza_patches

MAIN_CHECKOUT = Path("/home/theo/SCL/SCR/MorphoFavela")
RUN_ID = "wp02_horizon_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = MAIN_CHECKOUT / "runs" / RUN_ID


@pytest.fixture(scope="module")
def directions_weights():
    return generate_tregenza_patches()


@pytest.fixture(scope="module")
def sky():
    # EPW is gitignored data, present only in the main checkout, not this worktree.
    epw = MAIN_CHECKOUT / load_params()["weather"]["primary_epw"]
    if not epw.exists():
        pytest.skip(f"EPW not on disk: {epw}")
    return wp02_sky.build(epw)


def _flat_transform(cell: float, size: int):
    """A north-up Affine centred on (0, 0), `size` cells of `cell` metres on a side."""
    origin_x = -size * cell / 2
    origin_y = size * cell / 2
    return Affine(cell, 0, origin_x, 0, -cell, origin_y), origin_x, origin_y


def _surface_path(dtm, fps, cell_m, out_stem):
    surface_tif = build_surface(dtm, fps, cell_m, out_stem)
    is_building_tif = surface_tif.with_name(
        surface_tif.stem.replace("_surface", "_is_building") + ".tif"
    )
    return load_surface(surface_tif, is_building_tif)


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------

def test_flat_surface_identity(sky, directions_weights):
    directions, _weights = directions_weights
    cell, size = 1.0, 40
    transform, _ox, _oy = _flat_transform(cell, size)
    surface = np.zeros((size, size), dtype="float32")
    obs = np.array([[0.0, 0.0], [5.0, -3.0], [-10.0, 8.0]])

    vis, on_building = patch_visibility(
        surface, transform, obs, directions=directions, device="cpu",
        max_dist_m=50, step_m=cell,
    )

    assert not on_building.any()
    assert vis.all(), "flat surface must leave every patch visible"
    svf = sky.svf(vis.astype(float))
    assert svf == pytest.approx(np.ones(len(obs)), abs=1e-9)
    irr = sky.irradiation(vis.astype(float))
    assert irr == pytest.approx(np.full(len(obs), sky.patch_total_kwh.sum()), rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Infinite canyon, exact mask
# ---------------------------------------------------------------------------

def test_infinite_canyon_exact_mask(directions_weights):
    """Analytic mask (visible iff dz/|dy| > 2H/W) vs the engine, patch-for-patch.

    The allowance is the ray march's own positional quantum: a mismatched
    patch's analytic ratio dz/|dy| must land within one step_m of the
    threshold measured at the wall's distance (cell / (W/2)) — anything
    further off is a real defect, not discretization.
    """
    directions, _weights = directions_weights
    d = directions
    cell, size, W = 0.5, 400, 20.0
    quantum = cell / (W / 2)
    report = {}

    for hw in (0.25, 0.5, 1.0, 2.0, 3.0):
        H = hw * W
        transform, ox, oy = _flat_transform(cell, size)
        ys = oy - (np.arange(size) + 0.5) * cell
        xs = ox + (np.arange(size) + 0.5) * cell
        Y, _X = np.meshgrid(ys, xs, indexing="ij")
        surface = np.where(np.abs(Y) >= W / 2, H, 0.0).astype("float32")
        obs = np.array([[0.0, 0.0]])  # canyon floor, centreline

        vis, on_building = patch_visibility(
            surface, transform, obs, directions=directions, device="cpu",
            max_dist_m=100, step_m=cell, obs_height_m=0.0,
        )
        assert not on_building.any()
        vis = vis[0]

        with np.errstate(divide="ignore"):
            ratio = d[:, 2] / np.abs(d[:, 1])
            analytic = ratio > (2 * hw)

        mismatch = np.where(vis != analytic)[0]
        assert np.all(np.abs(ratio[mismatch] - 2 * hw) <= quantum), (
            f"H/W={hw}: patches {mismatch} disagree with the closed form outside "
            f"the one-step quantum {quantum:.4f}")
        report[hw] = int(len(mismatch))

    print("infinite canyon: mismatches within one-step quantum, per H/W:", report)


# ---------------------------------------------------------------------------
# 3. Isolated wall
# ---------------------------------------------------------------------------

def test_isolated_wall_shadow(directions_weights):
    """A finite-width wall at radius D, height H, spanning the +x hemisphere.

    Built as an annular arc (D <= r < D+thickness, x > 0) rather than a
    straight wall so the analytic prediction alt < atan(H/D) holds exactly
    for every azimuth in the span, independent of azimuth — a straight wall's
    horizon depends on cos(azimuth) too and would not give a clean closed
    form.
    """
    directions, _weights = directions_weights
    d = directions
    D, H = 30.0, 10.0
    cell, size = 0.5, 300
    transform, ox, oy = _flat_transform(cell, size)
    ys = oy - (np.arange(size) + 0.5) * cell
    xs = ox + (np.arange(size) + 0.5) * cell
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    thickness = 2 * cell
    surface = np.where((R >= D) & (R < D + thickness) & (X > 0), H, 0.0).astype("float32")
    obs = np.array([[0.0, 0.0]])

    vis, on_building = patch_visibility(
        surface, transform, obs, directions=directions, device="cpu",
        max_dist_m=100, step_m=cell, obs_height_m=0.0,
    )
    assert not on_building.any()
    vis = vis[0]

    alt = np.arcsin(np.clip(d[:, 2], -1.0, 1.0))
    thresh = np.arctan(H / D)
    tangent_tol = 1e-6
    excluded = np.abs(d[:, 0]) <= tangent_tol  # grazing the wall's edge: physically ambiguous
    span = d[:, 0] > tangent_tol
    expected = np.where(span, alt >= thresh, True)

    mismatch = np.where(vis[~excluded] != expected[~excluded])[0]
    assert len(mismatch) == 0, f"isolated wall: unexpected mismatches at indices {mismatch}"
    print(f"isolated wall: {int(excluded.sum())} tangent patch(es) excluded by definition")


# ---------------------------------------------------------------------------
# 4. Physical bounds and monotonicity
# ---------------------------------------------------------------------------

def test_bounds_and_monotone_in_building_height(sky, directions_weights):
    directions, weights = directions_weights
    cell, size = 1.0, 60
    transform, ox, oy = _flat_transform(cell, size)
    rng = np.random.default_rng(20260914)
    base_noise = rng.uniform(0.0, 1.0, size=(size, size)).astype("float32")

    b_rows, b_cols = slice(25, 29), slice(30, 34)
    obs_rows = rng.integers(0, size, 12)
    obs_cols = rng.integers(0, size, 12)
    xs = ox + (obs_cols + 0.5) * cell
    ys = oy - (obs_rows + 0.5) * cell
    obs = np.stack([xs, ys], axis=1)
    keep = ~((obs_rows >= 25) & (obs_rows < 29) & (obs_cols >= 30) & (obs_cols < 34))
    obs = obs[keep]

    prev_vis = None
    for Hb in (0.0, 3.0, 6.0, 12.0, 25.0):
        surface = base_noise.copy()
        surface[b_rows, b_cols] = Hb
        vis, on_building = patch_visibility(
            surface, transform, obs, directions=directions, device="cpu",
            max_dist_m=80, step_m=cell,
        )
        assert not on_building.any()

        for svf in (svf_unweighted(vis), svf_solid_angle(vis, weights), sky.svf(vis.astype(float))):
            assert ((svf >= -1e-9) & (svf <= 1 + 1e-9)).all()
        irr = sky.irradiation(vis.astype(float))
        assert (irr >= 0).all()
        assert (irr <= sky.patch_total_kwh.sum() + 1e-9).all()

        if prev_vis is not None:
            newly_opened = (~prev_vis) & vis
            assert not newly_opened.any(), (
                f"raising the building to {Hb} m opened a previously-blocked patch")
        prev_vis = vis


# ---------------------------------------------------------------------------
# Shared site-surface fixtures (used by tests 5 and 6)
# ---------------------------------------------------------------------------

_RDP_DTM = MAIN_CHECKOUT / "data/riodaspedras/dtm_extended_300m.tif"
_RDP_FOOTPRINTS = MAIN_CHECKOUT / "data/riodaspedras/buildings_extended_300m.gpkg"


@pytest.fixture(scope="module")
def riodaspedras_surface_1m():
    if not _RDP_DTM.exists() or not _RDP_FOOTPRINTS.exists():
        pytest.skip(f"site data not on disk: {_RDP_DTM} / {_RDP_FOOTPRINTS}")
    out_stem = RUN_DIR / "artifacts" / "riodaspedras_1m"
    surface, transform, _crs, is_building = _surface_path(_RDP_DTM, _RDP_FOOTPRINTS, 1.0, out_stem)
    return surface, transform, is_building


# ---------------------------------------------------------------------------
# 5. Device agreement
# ---------------------------------------------------------------------------

def test_device_agreement(riodaspedras_surface_1m, directions_weights):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    directions, _weights = directions_weights
    surface, transform, is_building = riodaspedras_surface_1m

    rng = np.random.default_rng(20260914)
    h, w = surface.shape
    rows = rng.integers(0, h, 6000)
    cols = rng.integers(0, w, 6000)
    on_ground = ~is_building[rows, cols]
    rows, cols = rows[on_ground][:2000], cols[on_ground][:2000]
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    obs = np.stack([np.asarray(xs), np.asarray(ys)], axis=1)

    vis_cpu, _ = patch_visibility(
        surface, transform, obs, directions=directions, is_building=is_building,
        device="cpu", max_dist_m=500.0, step_m=1.0,
    )
    vis_cuda, _ = patch_visibility(
        surface, transform, obs, directions=directions, is_building=is_building,
        device="cuda", max_dist_m=500.0, step_m=1.0,
    )
    mismatch_frac = float((vis_cpu != vis_cuda).mean())
    print(f"device agreement: {mismatch_frac:.5%} mismatched entries, n_observers={len(obs)}")
    assert mismatch_frac <= 0.001


# ---------------------------------------------------------------------------
# 6. CPU cross-reference (measured, then thresholded)
# ---------------------------------------------------------------------------

def _svf_variants(vis: np.ndarray, weights: np.ndarray, sky_obj) -> dict[str, np.ndarray]:
    return {
        "unweighted": svf_unweighted(vis),
        "solid_angle": svf_solid_angle(vis, weights),
        "cosine_weighted": sky_obj.svf(vis.astype(float)),
    }


def _compare(measured: np.ndarray, reference: np.ndarray) -> dict:
    delta = np.abs(measured - reference)
    r = float(np.corrcoef(measured, reference)[0, 1])
    return {
        "r": r,
        "median_abs_delta": float(np.median(delta)),
        "p95_abs_delta": float(np.percentile(delta, 95)),
        "max_abs_delta": float(np.max(delta)),
        "n": int(len(reference)),
    }


def test_cpu_crossreference_riodaspedras(sky, directions_weights):
    directions, weights = directions_weights
    ref_path = MAIN_CHECKOUT / "outputs/riodaspedras/svf_v2/svf_streets.gpkg"
    if not ref_path.exists():
        pytest.skip(f"reference gpkg not on disk: {ref_path}")
    if not _RDP_DTM.exists() or not _RDP_FOOTPRINTS.exists():
        pytest.skip(f"site data not on disk: {_RDP_DTM} / {_RDP_FOOTPRINTS}")

    ref = gpd.read_file(ref_path)
    # Spec: use the geometry's x/y (offset for 673/16905 points, up to ~3.3 m,
    # away from wall-flush original_x/original_y), not original_x/original_y.
    obs = np.column_stack([ref.geometry.x.to_numpy(), ref.geometry.y.to_numpy()])
    ref_svf = ref["svf"].to_numpy(dtype="float64")
    obs_height_m = float((ref["z_observer"] - ref["z"]).median())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict[str, dict] = {}
    variant_names: list[str] = []

    for cell_m in (1.0, 5.0):
        out_stem = RUN_DIR / "artifacts" / f"riodaspedras_{cell_m:g}m"
        surface, transform, _crs, is_building = _surface_path(_RDP_DTM, _RDP_FOOTPRINTS, cell_m, out_stem)

        vis, on_building = patch_visibility(
            surface, transform, obs, directions=directions, is_building=is_building,
            obs_height_m=obs_height_m, max_dist_m=500.0, step_m=cell_m, device=device,
        )
        variants = _svf_variants(vis, weights, sky)
        variant_names = list(variants)
        valid = ~on_building
        cell_result = {
            name: _compare(np.asarray(measured, dtype="float64")[valid], ref_svf[valid])
            for name, measured in variants.items()
        }
        cell_result["n_on_building_excluded"] = int(on_building.sum())
        results[f"{cell_m:g}m"] = cell_result

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RUN_DIR / "crossref_riodaspedras.json"
    out_json.write_text(json.dumps({
        "reference": str(ref_path),
        "n_points": int(len(ref)),
        "obs_height_m_measured": obs_height_m,
        "results": results,
    }, indent=1))
    print(json.dumps(results, indent=1))

    write_run_manifest(
        RUN_DIR,
        cell_m=1.0, obs_height_m=obs_height_m, max_dist_m=500.0, step_m=1.0,
        sampling_rule="16905 street points from outputs/riodaspedras/svf_v2/svf_streets.gpkg",
        device=device,
    )

    best_name = max(variant_names, key=lambda n: results["1m"][n]["r"])
    best = results["1m"][best_name]
    assert best["r"] >= 0.95 and best["median_abs_delta"] <= 0.03, (
        f"PROVISIONAL floor failed for best variant '{best_name}' at 1m: "
        f"r={best['r']:.4f}, median|delta|={best['median_abs_delta']:.4f} "
        f"(floor: r >= 0.95 and median|delta| <= 0.03) — measured numbers in {out_json}")
