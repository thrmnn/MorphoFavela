"""Data-independent unit tests for the vidigal patch-viz pure helpers.

Only the deterministic helpers are exercised (no TLS / synthetic data,
no QGIS / folium output): point→grid binning and disk∩hull overlap %.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cfd_overlay.build_vidigal_patch_viz import (  # noqa: E402
    bin_umag_to_grid,
    overlap_pct,
)


def test_overlap_pct_full_half_none():
    disk = Point(0, 0).buffer(50.0, quad_segs=128)
    assert overlap_pct(disk, box(-100, -100, 100, 100)) == pytest.approx(100.0, abs=1e-6)
    # right half-plane covers exactly half the disk
    assert overlap_pct(disk, box(0, -100, 100, 100)) == pytest.approx(50.0, abs=0.5)
    assert overlap_pct(disk, box(500, 500, 600, 600)) == 0.0


def test_bin_umag_to_grid_means_and_nodata():
    # four points: two share the SW cell (mean), others isolated; one cell empty
    x = np.array([0.0, 1.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    u = np.array([2.0, 4.0, 9.0, 1.0])
    arr, transform = bin_umag_to_grid(x, y, u, spacing=2.0)

    assert arr.dtype == np.float32
    valid = arr[np.isfinite(arr)]
    # points span 0..10 in both axes at 2 m → 6×6 grid
    assert arr.shape == (6, 6)
    # the two co-located-ish points (0,0) and (1,0) fall in the same 2 m
    # cell → mean (2+4)/2 = 3.0 ; isolated points keep their value
    assert 3.0 in np.round(valid, 6)
    assert 9.0 in np.round(valid, 6)
    assert 1.0 in np.round(valid, 6)
    # most cells are empty → NaN
    assert np.isnan(arr).sum() > arr.size / 2
    # north-up transform: origin half a pixel NW of the min point
    assert transform.c == pytest.approx(-1.0)   # x0 - spacing/2
    assert transform.f == pytest.approx(11.0)   # y_max + spacing/2
    assert transform.a == pytest.approx(2.0)
    assert transform.e == pytest.approx(-2.0)


def test_bin_umag_grid_row0_is_north():
    x = np.array([0.0, 0.0])
    y = np.array([0.0, 100.0])  # second point far north
    u = np.array([5.0, 7.0])
    arr, _ = bin_umag_to_grid(x, y, u, spacing=2.0)
    # north point must land in the top row, south point in the bottom row
    assert arr[0, 0] == pytest.approx(7.0)
    assert arr[-1, 0] == pytest.approx(5.0)
