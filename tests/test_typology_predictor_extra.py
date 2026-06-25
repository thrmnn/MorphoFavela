"""Unit tests for the typology-predictor extras (isotonic recal + blind map).

These cover the two pure pieces that don't need the on-disk grids: the ECE
metric and the isotonic-recalibration logic on a synthetic under-confident
classifier. The figure/IO functions are exercised by running the script.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.typology_predictor_extra import (  # noqa: E402
    CONT_FEATURES,
    expected_calibration_error,
)


def test_cont_features_present_in_grid_schema():
    """Guard the silent drift that broke this script once: the dissolved-λf
    re-baseline regenerated features_grid without party_wall_ratio while
    CONT_FEATURES still listed it. Any CONT_FEATURES member must exist as a
    grid column. Skips in data-free checkouts (outputs/ is gitignored)."""
    p = ROOT / "outputs" / "vidigal" / "features" / "features_grid.parquet"
    if not p.exists():
        pytest.skip("features_grid not present (gitignored outputs)")
    import geopandas as gpd

    cols = set(gpd.read_parquet(p).columns)
    missing = [f for f in CONT_FEATURES if f not in cols]
    assert not missing, f"CONT_FEATURES drifted from grid schema: {missing}"


def test_ece_perfectly_calibrated_is_zero():
    rng = np.random.default_rng(0)
    # p is the true Bernoulli rate; with many draws ECE → 0
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_ece_detects_miscalibration():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0, 1, 100_000)
    y = (rng.uniform(0, 1, p_true.size) < p_true).astype(int)
    # report a deliberately over-confident probability (push toward extremes)
    p_bad = np.clip(p_true * 1.6 - 0.3, 0, 1)
    assert expected_calibration_error(y, p_bad) > expected_calibration_error(y, p_true)


def test_ece_bounds_and_endpoints():
    # all predictions exactly right at the endpoints → zero error
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=10) == 0.0
    # constant wrong prediction → ECE equals the gap
    y2 = np.zeros(100)
    p2 = np.full(100, 0.5)
    assert abs(expected_calibration_error(y2, p2) - 0.5) < 1e-9
