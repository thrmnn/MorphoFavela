"""Unit tests for the T5a permutation power curve (scripts/health/tb_power_curve).

Pins the two properties that make the power bound meaningful: power lives in [0,1]
and rises with n. Tolerance-guarded against tiny-n permutation discreteness + MC
noise, so the checks are cheap (few trials) but still catch a broken generator."""

import numpy as np

from scripts.health.tb_power_curve import (
    min_n_for_power,
    perm_null_abs_rho,
    power_at,
    spearman_to_pearson,
)


def test_spearman_pearson_map_monotone_and_bounded():
    rs = [spearman_to_pearson(rho) for rho in (0.6, 0.7, 0.8, 0.9)]
    assert all(0 < r < 1 for r in rs)
    assert rs == sorted(rs)  # larger target ρ needs larger Pearson r


def test_power_in_unit_interval():
    rng = np.random.default_rng(1)
    for n in (5, 9, 14):
        p = power_at(n, 0.8, 400, rng, sorted_abs_null=perm_null_abs_rho(n, rng))
        assert 0.0 <= p <= 1.0


def test_power_increases_with_n():
    rng = np.random.default_rng(7)
    ns = [6, 10, 14, 18]
    powers = [power_at(n, 0.8, 800, rng, sorted_abs_null=perm_null_abs_rho(n, rng))
              for n in ns]
    # non-decreasing within MC tolerance, and a clear net rise across the range
    assert all(b >= a - 0.05 for a, b in zip(powers, powers[1:]))
    assert powers[-1] - powers[0] > 0.2


def test_tiny_n_cannot_reject():
    # n=4: exact min two-tailed p = 2/4! = 0.083 > 0.05 → power is exactly 0
    rng = np.random.default_rng(3)
    assert power_at(4, 0.9, 500, rng, sorted_abs_null=perm_null_abs_rho(4, rng)) == 0.0


def test_min_n_helper():
    assert min_n_for_power([4, 6, 8], [0.1, 0.5, 0.9], 0.8) == 8
    assert min_n_for_power([4, 6, 8], [0.1, 0.2, 0.3], 0.8) is None
