"""Unit tests for the T4 compound (sun × ventilation) exposure screen
(scripts/health/compound_exposure).

Pins the correlation plumbing on synthetic data (always runs) and, when the
per-site grids are present, the two invariants that make the built-cell
classification trustworthy: every share is a percentage in [0, 100] and the
compound (conjunction) share never exceeds either marginal deficit."""

import numpy as np
import pytest

from scripts.health.compound_exposure import (
    LAMBDA_F_PRIMARY,
    delta_sign_ci,
    site_deficits,
    three_rhos,
)
from scripts.health.tb_sun_deficit_screen import SITES, YEARS, incidence

ROOT_GRID = "outputs/{}/morphometrics/grid/grid_metrics.gpkg"


def test_three_rhos_bounded_and_monotone():
    tb = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sun = np.array([10.0, 20.0, 30.0, 40.0, 50.0])   # perfectly monotone with tb
    vent = np.array([50.0, 10.0, 40.0, 20.0, 30.0])  # scrambled
    comp = sun.copy()
    out = three_rhos(sun, vent, comp, tb)
    for k in ("sun", "ventilation", "compound"):
        assert -1.0 <= out[k]["rho"] <= 1.0
        assert 0.0 <= out[k]["perm_p_two_tailed"] <= 1.0
    assert out["sun"]["rho"] == pytest.approx(1.0)
    # perfect rank at n=5 → exact two-tailed perm p = 2/120
    assert out["sun"]["perm_p_two_tailed"] == pytest.approx(2 / 120, abs=1e-9)


def test_delta_sign_ci_well_formed():
    tb = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    a = tb.copy()                       # ρ(a,tb)=+1
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # ρ(b,tb)=-1
    lo, hi, frac = delta_sign_ci(a, b, tb, n=500)
    assert lo <= hi
    assert 0.0 <= frac <= 1.0
    assert frac > 0.5  # a beats b, so Δ=ρ(a)-ρ(b) is mostly positive


@pytest.mark.parametrize("site", list(SITES))
def test_site_deficits_invariants(site):
    from pathlib import Path

    from scripts.health.compound_exposure import ROOT

    if not (ROOT / ROOT_GRID.format(site)).exists():
        pytest.skip(f"grid data absent for {site}")
    d = site_deficits(site, LAMBDA_F_PRIMARY)
    assert d["n_built_cells"] > 0
    for k in ("sun_deficit_pct", "vent_deficit_pct", "compound_pct"):
        assert 0.0 <= d[k] <= 100.0
    # conjunction bound: cells failing BOTH ⊆ cells failing either marginal
    assert d["compound_pct"] <= d["sun_deficit_pct"] + 1e-9
    assert d["compound_pct"] <= d["vent_deficit_pct"] + 1e-9


def test_all_five_sites_yield_finite_rhos():
    from scripts.health.compound_exposure import ROOT

    sites = list(SITES)
    if not all((ROOT / ROOT_GRID.format(s)).exists() for s in sites):
        pytest.skip("grid data absent for one or more TB sites")
    tb = np.array([incidence(s, YEARS) for s in sites])
    sun, vent, comp = [], [], []
    for s in sites:
        dd = site_deficits(s, LAMBDA_F_PRIMARY)
        sun.append(dd["sun_deficit_pct"])
        vent.append(dd["vent_deficit_pct"])
        comp.append(dd["compound_pct"])
    out = three_rhos(sun, vent, comp, tb)
    assert len(sun) == 5
    for k in ("sun", "ventilation", "compound"):
        assert np.isfinite(out[k]["rho"])
        assert -1.0 <= out[k]["rho"] <= 1.0
