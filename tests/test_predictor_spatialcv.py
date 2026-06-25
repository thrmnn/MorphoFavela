"""Rank-1 spatial-CV hardening helpers (VIF collinearity audit)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("statsmodels")
from scripts.analyze_typology_predictor import CONT_FEATURES, vif_report  # noqa: E402


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
