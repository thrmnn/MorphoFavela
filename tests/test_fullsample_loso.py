"""Unit test for the full-sample LOSO robustness primitive.

Exercises ``loso_reduced`` on synthetic, cross-site-consistent separable data:
when a feature cleanly separates the classes the same way in every site, the
leave-one-site-out RF must recover near-perfect AUC and return one score per
held-out site in [0, 1].
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
    "run_fullsample_loso", ROOT / "scripts" / "run_fullsample_loso.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _synthetic(sites=("a", "b", "c"), n_per=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in sites:
        # class 0: feature ~ N(0,1); class 1: feature ~ N(3,1) — same rule per site
        for cls, mu in ((0, 0.0), (1, 3.0)):
            x = rng.normal(mu, 1.0, n_per)
            noise = rng.normal(0, 1.0, n_per)
            rows.append(pd.DataFrame({"feat": x, "noise": noise, "site": s, "sun_fail": cls}))
    return pd.concat(rows, ignore_index=True)


def test_loso_recovers_separable_signal():
    df = _synthetic()
    res = mod.loso_reduced(df, ["feat", "noise"], ["a", "b", "c"])
    assert set(res["per_site_auc"]) == {"a", "b", "c"}
    assert res["auc_mean"] > 0.9
    for a in res["per_site_auc"].values():
        assert 0.0 <= a <= 1.0


def test_loso_skips_single_class_site():
    # A site whose held-out fold has only one class cannot yield an AUC; it must
    # be skipped, not crash.
    df = _synthetic(sites=("a", "b"))
    only_zero = df[(df.site == "b")].copy()
    only_zero["sun_fail"] = 0
    df = pd.concat([df[df.site == "a"], only_zero], ignore_index=True)
    res = mod.loso_reduced(df, ["feat", "noise"], ["a", "b"])
    assert "b" not in res["per_site_auc"]
