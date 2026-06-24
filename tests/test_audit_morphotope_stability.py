"""Unit test for the morphotope bootstrap-stability audit primitives.

Exercises bootstrap_ari on a small synthetic, well-separated composition table:
a clean cluster structure must yield near-perfect ARI (refits recover the
reference labels), and the helper must return exactly N_BOOT scores in [-1, 1].
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "audit_morphotope_stability",
    ROOT / "scripts" / "audit_morphotope_stability.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

from src.morphometry.morphotope import N_TYPES, fit_morphotopes  # noqa: E402


def _synthetic_comp(n_per=200, k=3, seed=0):
    """k well-separated composition blobs on the simplex over comp_0..comp_5 +
    diversity — clean structure the GMM should recover on any 80% subsample."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(k):
        center = np.full(N_TYPES, 0.02)
        center[c] = 1.0 - 0.02 * (N_TYPES - 1)
        blob = center + rng.normal(0, 0.01, size=(n_per, N_TYPES))
        blob = np.clip(blob, 0, None)
        blob /= blob.sum(axis=1, keepdims=True)
        div = -(np.where(blob > 0, blob * np.log(blob + 1e-12), 0)).sum(1) / np.log(N_TYPES)
        rows.append(np.column_stack([blob, div]))
    X = np.vstack(rows)
    cols = [f"comp_{i}" for i in range(N_TYPES)] + ["diversity"]
    return pd.DataFrame(X, columns=cols)


def test_bootstrap_ari_shape_and_range():
    comp = _synthetic_comp()
    ref = fit_morphotopes(comp, k=3, random_state=0, n_init=2)
    aris = mod.bootstrap_ari(comp, ref, k=3)
    assert aris.shape == (mod.N_BOOT,)
    assert np.all(aris >= -1.0) and np.all(aris <= 1.0)


def test_bootstrap_ari_high_on_clean_structure():
    comp = _synthetic_comp()
    ref = fit_morphotopes(comp, k=3, random_state=0, n_init=2)
    aris = mod.bootstrap_ari(comp, ref, k=3)
    # well-separated blobs → refits recover the reference partition almost exactly
    assert aris.mean() > 0.95
    assert aris.min() > 0.80
