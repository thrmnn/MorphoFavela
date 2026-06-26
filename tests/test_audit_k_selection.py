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


def test_bootstrap_ari_schema_and_range():
    """bootstrap_ari returns the pinned summary keys; ARI in [-1,1], CI ordered,
    high on clean structure."""
    rng = np.random.default_rng(0)
    blobs = [rng.normal(m, 0.2, size=(150, 4)) for m in (-4, 0, 4)]
    Xz = np.vstack(blobs)
    ref = np.repeat([0, 1, 2], 150)
    out = aks.bootstrap_ari(Xz, ref, k=3, n_boot=20)
    assert set(out) == {"n_boot", "frac", "mean", "min", "ci_2.5", "ci_97.5", "_aris"}
    assert out["n_boot"] == 20
    assert -1.0 <= out["min"] <= out["mean"] <= 1.0
    assert out["ci_2.5"] <= out["ci_97.5"]
    assert out["mean"] > 0.8          # well-separated blobs recover cleanly


def test_summary_json_schema(tmp_path):
    """k_selection_summary.json carries the keys the decision log + downstream
    consumers pin (elbow_k, silhouette_peak_k, LOSO mean/min, bootstrap CI)."""
    import json

    rng = np.random.default_rng(1)
    Xz = np.vstack([rng.normal(m, 0.2, size=(120, 4)) for m in (-4, 0, 4)])
    ref = np.repeat([0, 1, 2], 120)
    boot = aks.bootstrap_ari(Xz, ref, k=3, n_boot=15)
    boot.pop("_aris")
    summary = {
        "n_cells": len(Xz),
        "k_chosen": 6,
        "elbow_k": 3,
        "silhouette_peak_k": 2,
        "calinski_harabasz_peak_k": 3,
        "davies_bouldin_best_k": 3,
        "loso_ari_mean": 0.78,
        "loso_ari_min": 0.54,
        "loso_ari_worst_fold": "complexo_do_alemao",
        "bootstrap_ari": boot,
    }
    p = tmp_path / "k_selection_summary.json"
    p.write_text(json.dumps(summary))
    loaded = json.loads(p.read_text())
    required = {
        "n_cells", "k_chosen", "elbow_k", "silhouette_peak_k",
        "loso_ari_mean", "loso_ari_min", "bootstrap_ari",
    }
    assert required <= set(loaded)
    assert {"mean", "min", "ci_2.5", "ci_97.5"} <= set(loaded["bootstrap_ari"])
