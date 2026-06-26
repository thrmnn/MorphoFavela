"""Shared morphometry invariants — built mask + phantom filter contracts.

Pins the two invariants other streams integrate against:
  - the canonical pooled built-cell population is n=64,389 (building_count>0
    over the 5 campaign sites) — and the signature-path mask reproduces it;
  - the phantom-tower detector drops topo==0 / altura≈base>40 m and is a no-op
    on clean / column-less schemas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.morphometry.invariants import (  # noqa: E402
    built_mask,
    drop_phantom_buildings,
    phantom_mask,
)

CAMPAIGN_SITES = ["vidigal", "rocinha", "riodaspedras", "complexo_do_alemao", "maré"]
POOLED_BUILT_N = 64_389


def _grid(building_count, lambda_p=None):
    d = {"building_count": building_count}
    if lambda_p is not None:
        d["lambda_p"] = lambda_p
    return pd.DataFrame(d)


def test_built_mask_canonical_is_building_count():
    g = _grid([0, 1, 3, 0], lambda_p=[0.0, 0.5, 0.9, 0.005])
    assert list(built_mask(g)) == [False, True, True, False]


def test_built_mask_lenient_adds_low_coverage_cells():
    g = _grid([0, 1, 0, 0], lambda_p=[0.0, 0.5, 0.02, 0.005])
    # cell index 2: building_count==0 but lambda_p>0.01 → lenient-only built
    assert list(built_mask(g, lenient=True)) == [False, True, True, False]
    assert list(built_mask(g, lenient=False)) == [False, True, False, False]


def test_phantom_mask_drops_topo_zero_and_altura_base():
    b = pd.DataFrame({
        "topo": [60.0, 0.0, 20.0, 55.0],
        "base": [50.0, 129.3, 12.0, 55.0],
        "altura": [10.0, 129.3, 8.0, 55.0],  # row 3: altura≈base, >40 m phantom
    })
    assert list(phantom_mask(b)) == [False, True, False, True]


def test_phantom_extruding_only_keeps_zero_height_topo_zero():
    b = pd.DataFrame({
        "topo": [0.0, 0.0],
        "base": [10.0, 0.0],
        "altura": [0.0, 5.0],  # row 0: topo==0 but altura<=0 → kept when extruding_only
    })
    assert list(phantom_mask(b, extruding_only=True)) == [False, True]
    assert list(phantom_mask(b, extruding_only=False)) == [True, True]


def test_phantom_noop_without_height_columns():
    b = pd.DataFrame({"foo": [1, 2]})
    assert not phantom_mask(b).any()
    assert len(drop_phantom_buildings(b)) == 2


def test_signature_path_built_mask_pooled_is_64389():
    """The signature-path built mask reproduces the pinned pooled population."""
    gpd = pytest.importorskip("geopandas")
    total = 0
    present = 0
    for s in CAMPAIGN_SITES:
        p = ROOT / "outputs" / s / "features" / "features_grid.parquet"
        if not p.exists():
            continue
        present += 1
        g = pd.DataFrame(gpd.read_parquet(p).drop(columns="geometry"))
        total += int(built_mask(g).sum())
    if present < len(CAMPAIGN_SITES):
        pytest.skip("campaign feature grids absent; cannot check pooled n")
    assert total == POOLED_BUILT_N
