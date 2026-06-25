"""Unit test for the λf dissolve primitive (party-wall removal).

``dissolved_lambda_f`` must treat touching (party-walled) footprints as one
physical block — so two parcels split by a party wall give the SAME frontal area
as the single merged block, not the (inflated) sum — while genuinely separated
blocks still contribute independently.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "prototype_lambda_f_dissolve", ROOT / "scripts" / "prototype_lambda_f_dissolve.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

CELL = box(0, 0, 10, 10)
AREA = 100.0


def test_party_wall_removed():
    # One 10×10 block, height 10, vs the same block split by a party wall at
    # y=5 into two touching parcels. Dissolved λf must be identical — the
    # internal wall is not frontal area.
    single = mod.dissolved_lambda_f(CELL, AREA, [box(0, 0, 10, 10)], [10.0])
    split = mod.dissolved_lambda_f(
        CELL, AREA, [box(0, 0, 10, 5), box(0, 5, 10, 10)], [10.0, 10.0]
    )
    assert abs(single - split) < 1e-9
    assert single > 0


def test_dissolved_below_naive_sum():
    # The naive per-parcel sum for the split case double-counts the cross-wind
    # extent (both halves span the full width on the x-axis), so dissolved must
    # be strictly less than that sum.
    split = mod.dissolved_lambda_f(
        CELL, AREA, [box(0, 0, 10, 5), box(0, 5, 10, 10)], [10.0, 10.0]
    )
    # naive sum = 2× the single-block value on the axis where both halves overlap
    single = mod.dissolved_lambda_f(CELL, AREA, [box(0, 0, 10, 10)], [10.0])
    assert split < 1.9 * single  # nowhere near the 2× a naive sum would give


def test_separated_blocks_sum_linearly():
    # Two non-touching identical slivers are distinct roughness elements: their
    # frontal areas add (no merging, no sheltering correction), so the pair is
    # exactly twice a single sliver.
    one = mod.dissolved_lambda_f(CELL, AREA, [box(0, 0, 2, 10)], [10.0])
    two = mod.dissolved_lambda_f(
        CELL, AREA, [box(0, 0, 2, 10), box(8, 0, 10, 10)], [10.0, 10.0]
    )
    assert one > 0
    assert abs(two - 2 * one) < 1e-9
