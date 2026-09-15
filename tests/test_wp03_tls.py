"""WP-03 acceptance tests. Spec: docs/wp03_tls_spec.md, deliverable 3.

Large data (e57 scans, DTM, footprints) lives only in the main checkout;
data/ and outputs/ are gitignored and were never copied into this worktree.
Pure-function tests (alley-width classing, G2 floor selection, identical-
surface delta=0) need no data at all. Run-output tests read the most recent
runs/wp03_tls_*/ folder under REPO_ROOT (this worktree writes its own run
outputs there, per the spec) and skip cleanly if none exists yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

import rasterio.transform
import rasterio.warp

from src.brisa_solar.constants import P1_SKY_PATCHES, REPO_ROOT
from src.brisa_solar.wp03_tls import (
    ALLEY_CLASSES,
    alley_width_class,
    als_fill_surface,
    compute_svf_pair,
    coverage_share_mask,
    g2_floor,
    _resample_array_to_grid,
    _shift_transform,
)
from src.svf_v2.compute import generate_tregenza_patches


# ---------------------------------------------------------------------------
# (a) alley-width classing on a synthetic footprint pair reproduces known widths
# ---------------------------------------------------------------------------

def test_alley_width_class_known_gap():
    # Two half-planes with a 2 m gap between them (edges at x=-1, x=1): a point
    # on the centreline is 1 m from each edge -> width = 2*1 = 2 m -> "1.5-3m".
    left = box(-20, -20, -1, 20)
    right = box(1, -20, 20, 20)
    gdf = gpd.GeoDataFrame(geometry=[left, right])
    dist, labels = alley_width_class(np.array([[0.0, 0.0]]), gdf)
    assert dist[0] == pytest.approx(1.0)
    assert labels[0] == "1.5-3m"


def test_alley_width_class_narrow_and_wide_buckets():
    gdf_narrow = gpd.GeoDataFrame(geometry=[box(-20, -20, -0.4, 20), box(0.4, -20, 20, 20)])
    _dist_n, labels_n = alley_width_class(np.array([[0.0, 0.0]]), gdf_narrow)
    assert labels_n[0] == "<1.5m"

    gdf_wide = gpd.GeoDataFrame(geometry=[box(-20, -20, -5, 20), box(5, -20, 20, 20)])
    _dist_w, labels_w = alley_width_class(np.array([[0.0, 0.0]]), gdf_wide)
    assert labels_w[0] == ">3m"


# ---------------------------------------------------------------------------
# (b) G2 floor selection rule on a synthetic class table
# ---------------------------------------------------------------------------

def test_g2_floor_picks_narrowest_passing_class():
    table = [
        {"class": "<1.5m", "median_abs_delta": 0.25},
        {"class": "1.5-3m", "median_abs_delta": 0.07},
        {"class": ">3m", "median_abs_delta": 0.01},
    ]
    assert g2_floor(table) == "1.5-3m"


def test_g2_floor_none_when_no_class_passes():
    table = [{"class": c, "median_abs_delta": 0.5} for c in ALLEY_CLASSES]
    assert g2_floor(table) is None


def test_g2_floor_ignores_class_order_and_handles_missing_classes():
    table = [{"class": ">3m", "median_abs_delta": 0.02}]
    assert g2_floor(table) == ">3m"


# ---------------------------------------------------------------------------
# (c) the same engine call on identical rasters gives Delta = 0 exactly
# ---------------------------------------------------------------------------

def test_identical_surfaces_give_exact_zero_delta():
    directions, weights = generate_tregenza_patches()
    cell, size = 1.0, 40
    transform = Affine(cell, 0, -size * cell / 2, 0, -cell, size * cell / 2)
    rng = np.random.default_rng(0)
    surface = rng.uniform(0, 5, size=(size, size)).astype("float32")
    obs = np.array([[0.0, 0.0], [5.0, -3.0], [-8.0, 6.0]])

    svf_a, svf_b = compute_svf_pair(
        surface, transform, cell, surface, transform, cell,
        obs, directions, weights, obs_height_m=1.5, max_dist_m=100.0, device="cpu",
    )
    assert np.all(svf_a - svf_b == 0.0)


# ---------------------------------------------------------------------------
# WP-03B deliverable 5(a): ALS-fill leaves TLS-covered cells untouched and
# sets uncovered cells to the (resampled) ALS surface value.
# ---------------------------------------------------------------------------

def test_als_fill_leaves_covered_cells_untouched_and_fills_uncovered_with_als():
    transform = Affine(1.0, 0, 0, 0, -1.0, 10)
    tls_dsm = np.full((5, 5), -9999.0, dtype="float32")
    tls_dsm[1:3, 1:3] = 7.0  # a small covered patch
    als_surface = (np.arange(25, dtype="float32").reshape(5, 5) + 100.0)

    filled = als_fill_surface(tls_dsm, transform, "EPSG:31983", als_surface, transform, "EPSG:31983")

    covered = tls_dsm != -9999.0
    assert np.array_equal(filled[covered], tls_dsm[covered])
    assert np.allclose(filled[~covered], als_surface[~covered])


# ---------------------------------------------------------------------------
# WP-03B deliverable 5(b): the covered-only filter keeps exactly the
# observers whose march-disc coverage share meets the threshold.
# ---------------------------------------------------------------------------

def test_coverage_share_mask_keeps_observers_meeting_threshold():
    transform = Affine(1.0, 0, 0, 0, -1.0, 10)
    coverage_share = np.array([[1.0, 0.9], [0.5, 0.79]])
    rows = [0, 0, 1, 1]
    cols = [0, 1, 0, 1]
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    obs_xy = np.column_stack([xs, ys])

    keep = coverage_share_mask(obs_xy, transform, coverage_share, min_share=0.8)

    assert list(keep) == [True, True, False, False]


# ---------------------------------------------------------------------------
# WP-03B deliverable 5(c): the shift corrector moves a synthetic raster by
# exactly (dx, dy) cells.
# ---------------------------------------------------------------------------

def test_shift_transform_moves_raster_by_exact_integer_cells():
    transform = Affine(1.0, 0, 0, 0, -1.0, 10)
    src = np.arange(100, dtype="float64").reshape(10, 10)
    dx_cells, dy_cells = 2, 3

    shifted_transform = _shift_transform(transform, dx_cells, dy_cells)
    out = _resample_array_to_grid(
        src, transform, "EPSG:31983", shifted_transform, src.shape, "EPSG:31983",
        resampling=rasterio.warp.Resampling.nearest,
    )

    h, w = src.shape
    expected = np.full(src.shape, np.nan)
    for r in range(h):
        for c in range(w):
            sr, sc = r + dy_cells, c + dx_cells
            if 0 <= sr < h and 0 <= sc < w:
                expected[r, c] = src[sr, sc]

    valid = np.isfinite(expected)
    assert valid.sum() > 0
    assert np.allclose(out[valid], expected[valid])
    assert np.all(np.isnan(out[~valid]))


# ---------------------------------------------------------------------------
# (d) manifest carries sky_patches == P1_SKY_PATCHES, PDAL version, point counts
# (e) run-output tests skip cleanly when the run folder is absent
# ---------------------------------------------------------------------------

def _latest_run_dir() -> Path | None:
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        p for p in runs_dir.glob("wp03_tls_2*")  # excludes wp03_tls_probe_*
        if p.is_dir()
    )
    return candidates[-1] if candidates else None


def test_manifest_carries_sky_pdal_version_and_point_counts():
    run_dir = _latest_run_dir()
    if run_dir is None:
        pytest.skip("no runs/wp03_tls_<UTC>/ output yet")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"{manifest_path} not written yet")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sky"]["patches"] == P1_SKY_PATCHES
    assert manifest.get("pdal_version")
    assert manifest.get("point_counts")
    assert all(v for v in manifest["point_counts"].values())
