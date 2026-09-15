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

from src.brisa_solar.constants import P1_SKY_PATCHES, REPO_ROOT
from src.brisa_solar.wp03_tls import (
    ALLEY_CLASSES,
    alley_width_class,
    compute_svf_pair,
    g2_floor,
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
