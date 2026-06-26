"""Rank-1 spatial-CV hardening helpers (VIF collinearity audit)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("statsmodels")
pytest.importorskip("sklearn")
from scripts.analyze_typology_predictor import CONT_FEATURES, loso, vif_report  # noqa: E402
from src.morphometry.signature import CAMPAIGN_SITES  # noqa: E402


def test_vif_flags_a_collinear_pair_and_spares_independents():
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame({f: rng.normal(size=n) for f in CONT_FEATURES})
    shared = rng.normal(size=n)
    df[CONT_FEATURES[0]] = shared
    df[CONT_FEATURES[1]] = shared + rng.normal(scale=1e-2, size=n)  # near-duplicate
    v = vif_report(df)
    assert v[CONT_FEATURES[0]] > 10 and v[CONT_FEATURES[1]] > 10  # collinear pair flagged
    assert v[CONT_FEATURES[3]] < 5  # an independent feature stays low


def _flip_frame(seed=0):
    """Synthetic LOSO frame where a continuous feature cleanly separates the
    binary failure target but the type/morphotope codes are pure noise wrt it.
    Pins the headline vector>type FLIP direction against future drift."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in CAMPAIGN_SITES:
        n = 400
        z = rng.normal(size=n)  # the only real driver of failure
        d = pd.DataFrame({
            "site": s,
            "fail": (z + rng.normal(scale=0.3, size=n) > 0).astype(int),
            "morphotype_smooth": rng.integers(0, 4, size=n),   # noise wrt fail
            "morphotope": rng.integers(0, 3, size=n),          # noise wrt fail
        })
        d[CONT_FEATURES[0]] = z  # continuous signal carrier
        for f in CONT_FEATURES[1:]:
            d[f] = rng.normal(size=n)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def test_vector_beats_type_on_clean_separation():
    """Exercises the real spatially-blocked LOSO path: continuous fabric vector
    must transfer (high AUC-PR) where the type code cannot. Guards the FLIP."""
    df = _flip_frame()
    auc_type = loso(df, "type")[0]["auc_pr"].mean()
    auc_vector = loso(df, "vector")[0]["auc_pr"].mean()
    assert 0.0 <= auc_type <= 1.0
    assert 0.0 <= auc_vector <= 1.0
    assert auc_vector > auc_type
