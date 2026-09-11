"""WP-02 acceptance: analytic self-checks for the cumulative sky.

These replace the plan's r.sun cross-check (GRASS is not installable here) and
are stronger than it would have been: r.sun could share a bug class with any
other implementation, whereas an identity and a closed-form canyon cannot.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.brisa_solar import wp02_sky
from src.brisa_solar.constants import REPO_ROOT, P1_SKY_PATCHES, load_params


@pytest.fixture(scope="module")
def sky():
    epw = REPO_ROOT / load_params()["weather"]["primary_epw"]
    if not epw.exists():
        pytest.skip(f"EPW not on disk: {epw}")
    return wp02_sky.build(epw)


def test_hemisphere_solid_angle(sky):
    """The patch weights must tile the hemisphere exactly."""
    assert sky.weights.sum() == pytest.approx(2 * np.pi, rel=1e-6)


def test_cosine_projection_bias_is_known_and_bounded(sky):
    """Centre-point cosines over-count the projection; pin the size of that error.

    Measured 0.548%. It is normalised out of the energy distribution, but if the
    patch scheme ever changes this test says so rather than letting a new bias
    ride silently into every irradiation number.
    """
    bias = (sky.cosine_sum - np.pi) / np.pi
    assert 0.0 < bias < 0.01, f"cosine-projection bias {bias:.4%} outside the known band"


def test_unobstructed_identity_reproduces_the_epw(sky):
    """A2.1 — an unobstructed cell must receive the EPW's own annual energy.

    Compared against the EPW's COMPONENTS (DHI + DNI*cos z), not its reported
    GHI: those differ by 0.95% in this file, which is the file's internal
    inconsistency and not something the model should absorb or hide.
    """
    total = float(sky.patch_total_kwh.sum())
    assert total == pytest.approx(
        sky.patch_diffuse_kwh.sum() + sky.patch_direct_kwh.sum(), rel=1e-12)
    # within 2% of reported GHI (the plan's tolerance), and the residual is the
    # file's closure error — asserted as a bound, not a magic equality.
    assert abs(total - sky.annual_ghi_kwh) / sky.annual_ghi_kwh < 0.02


def test_infinite_canyon_svf_matches_the_closed_form(sky):
    """A2.2 — Oke's infinite-canyon SVF, from geometry alone.

    Canyon runs east-west, walls of height H at +-W/2. A ray clears the wall
    iff dz/|dy| > 2H/W, and the closed form is 1/sqrt(1+(2H/W)^2).

    The tolerance is MEASURED, not chosen: binary patch-centre visibility on a
    145-patch sky resolves a boundary only to ~11 degrees, and a sweep over
    H/W in {0.25 .. 3.0} bounds that discretization error at 0.033 absolute
    (see test_deep_canyons_are_the_discretization_worst_case). 0.04 catches a
    real regression — a sign error, a swapped axis, a broken weight — without
    failing on a known limit of the scheme.
    """
    d = sky.directions
    for hw in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        expected = 1 / np.sqrt(1 + (2 * hw) ** 2)
        with np.errstate(divide="ignore"):
            visible = (d[:, 2] / np.abs(d[:, 1])) > (2 * hw)
        got = float(sky.svf(visible.astype(float)))
        assert got == pytest.approx(expected, abs=0.04), (
            f"H/W={hw}: canyon SVF {got:.4f} vs closed form {expected:.4f}")


def test_deep_canyons_are_the_discretization_worst_case(sky):
    """The error is not uniform, and it leans the wrong way for this paper.

    Patch-centre visibility OVER-estimates sky access in deep canyons: at
    H/W = 3 the closed form is 0.1644 and the discretised answer is 0.1972,
    +20% relative. Favela alleys are exactly that geometry, and the citywide
    claim is sharpest where canyons are narrowest — so this bias is recorded
    here, must be stated in the methods, and is the quantitative reason the G2
    alley-width validity floor exists rather than being a formality.
    """
    d = sky.directions
    deep, shallow = 3.0, 0.25
    out = {}
    for hw in (shallow, deep):
        with np.errstate(divide="ignore"):
            vis = (d[:, 2] / np.abs(d[:, 1])) > (2 * hw)
        out[hw] = float(sky.svf(vis.astype(float))) - 1 / np.sqrt(1 + (2 * hw) ** 2)
    assert out[deep] > 0, "deep-canyon bias is expected POSITIVE (over-estimates sky)"
    assert abs(out[deep]) > abs(out[shallow]), (
        "the deep canyon must remain the worst case; if this flips, the sky "
        "scheme changed and every narrow-alley number needs re-checking")


def test_physical_bounds(sky):
    """A2.5 — nothing may see more sky, or receive more energy, than the open sky."""
    open_sky = np.ones(P1_SKY_PATCHES)
    assert sky.svf(open_sky) == pytest.approx(1.0, abs=1e-9)
    assert sky.svf(np.zeros(P1_SKY_PATCHES)) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(20260908)
    vis = rng.integers(0, 2, size=(50, P1_SKY_PATCHES)).astype(float)
    irr = sky.irradiation(vis)
    assert (irr >= 0).all()
    assert (irr <= float(sky.patch_total_kwh.sum()) + 1e-9).all()
    svfs = sky.svf(vis)
    assert ((svfs >= 0) & (svfs <= 1)).all()


def test_energy_is_monotone_in_visibility(sky):
    """Opening one more patch can never reduce annual irradiation."""
    base = np.zeros(P1_SKY_PATCHES)
    prev = sky.irradiation(base[None, :])[0]
    for i in range(P1_SKY_PATCHES):
        base[i] = 1.0
        cur = sky.irradiation(base[None, :])[0]
        assert cur >= prev - 1e-12
        prev = cur
