"""Smoke + invariant tests for the k-selection audit helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "audit_k_selection", ROOT / "scripts" / "audit_k_selection.py"
)
aks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aks)


def test_bic_elbow_returns_k_in_range():
    import pandas as pd

    sel = pd.DataFrame({
        "k": list(range(2, 11)),
        # convex-ish decreasing curve with a knee near k=4
        "bic": [100, 70, 55, 50, 48, 47, 46.5, 46.2, 46.0],
    })
    k = aks.bic_elbow_k(sel)
    assert 2 <= k <= 10


def test_match_labels_recovers_permutation():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 4, size=400)
    # candidate is a fixed relabeling of ref (perfect overlap, permuted ids)
    perm = {0: 2, 1: 3, 2: 0, 3: 1}
    cand = np.array([perm[c] for c in ref])
    matched = aks.match_labels(ref, cand, k=4)
    assert np.array_equal(matched, ref)


def test_match_labels_is_permutation_only():
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 5, size=300)
    cand = rng.integers(0, 5, size=300)
    matched = aks.match_labels(ref, cand, k=5)
    # matching only relabels, never invents labels outside 0..k-1
    assert set(np.unique(matched)).issubset(set(range(5)))
    assert len(matched) == len(cand)
