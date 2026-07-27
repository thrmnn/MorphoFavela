"""T5a — permutation power curve for the TB × winter-sun-deficit screen.

Bounds the STATISTICAL POWER of the tb_sun_deficit_screen probe. That screen sees
a headline Spearman ρ≈+0.80 across n=5 favela-bairros but an exact two-tailed
permutation p≈0.13 — NOT significant. The obvious question is whether n=5 could
ever have detected a real effect this size. This script answers it: for a family
of true population correlations ρ ∈ {0.6, 0.7, 0.8, 0.9}, Monte-Carlo simulate
paired rank data at each n and measure the fraction of trials the exact/sampled
permutation test rejects at α=0.05.

Method: draw a bivariate normal with Pearson r set so the induced Spearman equals
the target (ρ ≈ (6/π)·arcsin(r/2) ⇒ r = 2·sin(π·ρ/6); realized ρ recorded in the
JSON as a check), rank both coords, and score the exact two-tailed permutation p
(same |ρ|≥|ρ₀| rule as tb_sun_deficit_screen.exact_perm_p). The permutation null
depends ONLY on n — ranks are always {0..n-1} — so it is precomputed once per n
(exact enumeration for n≤8, else a fixed sampled null) instead of per trial.

HONEST CAVEAT: this bounds power, it does not rescue the screen. n=5 sits far below
the ~80%-power threshold for ρ=0.8, and at n≤4 the exact test CANNOT reach p<0.05
at all (min two-tailed p = 2/4! = 0.083). The screen stays exploratory/direction-
only; this curve just quantifies how underpowered it is.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "comparative" / "health"

NS = list(range(4, 25))
RHOS = [0.6, 0.7, 0.8, 0.9]
N_TRIALS = 3000
N_PERM_SAMPLED = 20000  # sampled permutation null when n! is too large (n>8)
ALPHA = 0.05
MAX_EXACT_N = 8
SEED = 20260727


def spearman_to_pearson(rho: float) -> float:
    """Pearson r of a bivariate normal whose induced Spearman ρ equals `rho`."""
    return 2.0 * np.sin(np.pi * rho / 6.0)


def _rho_from_dot(s, n: int):
    """Spearman ρ from the rank dot-product s=Σ xr·yr (ranks 0..n-1, no ties)."""
    sq = (n - 1) * n * (2 * n - 1) / 6.0  # Σ i² for i in 0..n-1
    return 1.0 - 12.0 * (sq - np.asarray(s)) / (n * (n**2 - 1))


def perm_null_abs_rho(n: int, rng) -> np.ndarray:
    """Sorted |ρ| under the permutation null for sample size n (exact if n≤8)."""
    base = np.arange(n)
    if n <= MAX_EXACT_N:
        perms = np.array(list(permutations(range(n))))
        dots = perms @ base
    else:
        dots = np.array([base @ rng.permutation(n) for _ in range(N_PERM_SAMPLED)])
    return np.sort(np.abs(_rho_from_dot(dots, n)))


def power_at(n: int, rho: float, n_trials: int, rng, alpha: float = ALPHA,
             sorted_abs_null: np.ndarray | None = None) -> float:
    """Fraction of Monte-Carlo trials rejecting H0 at `alpha` (exact perm test)."""
    if sorted_abs_null is None:
        sorted_abs_null = perm_null_abs_rho(n, rng)
    m = len(sorted_abs_null)
    r = spearman_to_pearson(rho)
    cov = np.array([[1.0, r], [r, 1.0]])
    hits = 0
    for _ in range(n_trials):
        xy = rng.multivariate_normal([0.0, 0.0], cov, size=n)
        xr = np.argsort(np.argsort(xy[:, 0]))
        yr = np.argsort(np.argsort(xy[:, 1]))
        rho0 = _rho_from_dot(xr @ yr, n)
        idx = np.searchsorted(sorted_abs_null, abs(rho0) - 1e-9, side="left")
        if (m - idx) / m < alpha:
            hits += 1
    return hits / n_trials


def realized_spearman(n: int, rho: float, rng, n_check: int = 4000) -> float:
    """Mean empirical Spearman ρ of the generator — sanity check on the r→ρ map."""
    r = spearman_to_pearson(rho)
    cov = np.array([[1.0, r], [r, 1.0]])
    vals = [spearmanr(*rng.multivariate_normal([0.0, 0.0], cov, size=n).T)[0]
            for _ in range(n_check)]
    return float(np.mean(vals))


def min_n_for_power(ns, powers, target: float = 0.80):
    """Smallest n reaching `target` power, or None if the grid never does."""
    for n, p in zip(ns, powers):
        if p >= target:
            return n
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    null_by_n = {n: perm_null_abs_rho(n, rng) for n in NS}

    grid = {rho: [power_at(n, rho, N_TRIALS, rng, sorted_abs_null=null_by_n[n])
                  for n in NS] for rho in RHOS}
    realized = {rho: realized_spearman(NS[-1], rho, rng) for rho in RHOS}
    min_n = {rho: min_n_for_power(NS, grid[rho]) for rho in RHOS}
    n5 = {rho: grid[rho][NS.index(5)] for rho in RHOS}

    print("=== T5a permutation power curve (α=0.05, exact≤n8 / sampled>n8) ===")
    print(f"  trials/point={N_TRIALS}  n-grid {NS[0]}..{NS[-1]}  ρ={RHOS}")
    for rho in RHOS:
        mn = min_n[rho]
        print(f"  ρ={rho:.1f}: power@n5={n5[rho]:.2f}  "
              f"min-n@80%={mn if mn else '>%d' % NS[-1]}  "
              f"(realized ρ≈{realized[rho]:+.2f})")
    mn8 = min_n[0.8]
    print(f"\n  HEADLINE: min n for 80% power at ρ=0.8, α=0.05 = "
          f"{mn8 if mn8 else '>%d' % NS[-1]}  (the screen has n=5)")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    colors = {0.6: "#c9b18a", 0.7: "#c98a4b", 0.8: "#b5651d", 0.9: "#7a3d0c"}
    for rho in RHOS:
        ax.plot(NS, grid[rho], "-o", ms=4, lw=1.6, color=colors[rho],
                label=f"true ρ = {rho:.1f}")
    ax.axhline(0.80, ls="--", c="#444", lw=1.0)
    ax.text(NS[-1], 0.80, " 80% power", va="bottom", ha="right", fontsize=8, color="#444")
    ax.axvline(5, ls=":", c="#900", lw=1.2)
    ax.text(5, 0.02, " screen n=5", rotation=90, va="bottom", ha="left",
            fontsize=8, color="#900")
    if mn8:
        ax.axvline(mn8, ls=":", c="#b5651d", lw=1.0, alpha=0.7)
    ax.set_xlabel("sample size  n  (favela-bairros)")
    ax.set_ylabel("power  (P[exact permutation p < 0.05])")
    ax.set_title("Permutation-test power vs n — TB × sun-deficit screen (T5a)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(NS[::2])
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.text(0.02, 0.97,
            f"min n for 80% power at ρ=0.8: {mn8 if mn8 else '>'+str(NS[-1])}\n"
            "exact two-tailed permutation test (|ρ|≥|ρ₀|); n≤4 can never reach p<0.05\n"
            "the screen's n=5 is far below the powered regime — direction-only stands",
            transform=ax.transAxes, va="top", fontsize=8.3, color="#555",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f2ec", ec="#e0d6c8"))
    fig.tight_layout()
    png = OUT / "tb_power_curve.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"  figure → {png.relative_to(ROOT)}")

    (OUT / "tb_power_curve.json").write_text(json.dumps({
        "framing": "Monte-Carlo power bound for the exact two-tailed permutation "
                   "Spearman test used by tb_sun_deficit_screen; the screen's n=5 is "
                   "far below the powered regime — this quantifies how underpowered.",
        "alpha": ALPHA, "n_trials": N_TRIALS, "n_grid": NS, "target_rhos": RHOS,
        "perm_null": {"exact_max_n": MAX_EXACT_N, "sampled_count": N_PERM_SAMPLED},
        "seed": SEED,
        "power_grid": {f"{rho:.1f}": [round(p, 4) for p in grid[rho]] for rho in RHOS},
        "power_at_n5": {f"{rho:.1f}": round(n5[rho], 4) for rho in RHOS},
        "min_n_for_80pct_power": {f"{rho:.1f}": min_n[rho] for rho in RHOS},
        "min_n_for_80pct_power_rho08": min_n[0.8],
        "realized_spearman_check": {f"{rho:.1f}": round(realized[rho], 4) for rho in RHOS},
        "caveat": "n≤4 cannot reach p<0.05 (min exact two-tailed p = 2/4! = 0.083); "
                  "tiny-n discreteness makes the curve step, not smooth.",
    }, indent=2, ensure_ascii=False))
    print(f"  json   → {(OUT / 'tb_power_curve.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
