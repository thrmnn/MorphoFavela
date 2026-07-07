"""First REAL health-data screen: TB incidence vs modelled winter-sun deficit.

Ecological, hypothesis-generating ONLY. Pairs our physically-modelled winter
sun-deficit exposure (share of street observers below the WHO 2 h/day direct-sun
floor) against REAL tuberculosis incidence for the favela≈bairro sites where a
bairro is essentially a single favela. This is the exposure→outcome step the
planetary-health track pivoted to (vitamin D is the mechanism, TB the measurable
endpoint; docs/planetary_health_plan.md).

DO NOT read a causal claim into this. n is tiny (4–5 favelas), the join is
ecological (bairro ≠ favela exactly), TB confounds with income/crowding/HIV, and
— demonstrated below — the association is support-dependent: it is visible at the
bairro≈favela scale and WASHES OUT at the coarser Área-de-Planejamento (AP) scale
(the change-of-support / MAUP caveat, live).

PROVENANCE (every number sourced; retrieved 2026-07-06):
- TB new-case counts by bairro de residência, FULL 2015–2023 series (tubeYY.dbf):
  SMS-Rio SINAN TabNet, tuberc2007.def, semicolon 'prn' export. See TB_YEARLY below;
  the 9-yr mean is the headline (kills small-count noise, esp. Vidigal 24–63/yr and the
  Alemão bairro-split artefact). Recipe (scripts cleanly): POST cgi-bin/tabnet?…tuberc2007.def
  with latin-1 params (Coluna=--N%E3o-Ativa--), strip NUL bytes, parse 5-dot bairro rows.
- STATS CAVEAT: at n=4–5 the parametric Spearman p is unreliable (t-approximation → p≈0 for
  ρ=1). The honest evidence is the bootstrap 95% CI and the fraction-of-specs-positive, NOT p.
- IBGE Censo 2022 favela/comunidade populations (bairro ≈ favela for these sites):
  Rocinha 72,021 · Vidigal 15,112 · Maré 124,832 · Alemão 54,202. Jacarezinho: 2010
  census bairro 37,839 (2022 not separately confirmed this pass — flagged; Spearman
  is rank-robust to it, TB rate stays 'high' for any pop in 30–50k).
- AP-level TB incidence /100k (published): EpiRio Boletim TB 2024, Tab.1 — AP2.1
  98.0/108.9, AP3.1 103.9/113.4, AP4.0 67.6/72.5 (2022/2023).
- Historical cross-check: Pereira et al., Rev. Saúde Pública 2015 (PMC4544397),
  Rocinha crude 447.3/100k in 2004–06 → our ~490 in 2022–23 (persisted, intensified).
- Sun-deficit: recomputed from outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg
  (share of observers with solar_hours_winter < 2.0 h).

⚠ Complexo do Alemão's bairro count collapses 2020→2022 (76→16→37) — a notification/
geocoding artefact of the recent bairro split, not a real drop. It is shown but
excluded from the primary correlation; its AP3.1 rate (~108) is the safer figure.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260706)  # deterministic bootstrap
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "comparative" / "health"

YEARS = list(range(2015, 2024))
# TB new cases by bairro de residência, 2015–2023 — SMS-Rio SINAN TabNet
# (tuberc2007.def, tubeYY.dbf; retrieved 2026-07-06, provenance in module docstring).
TB_YEARLY = {
    "rocinha":            [337, 273, 274, 255, 207, 253, 292, 351, 354],
    "vidigal":            [33, 31, 24, 27, 36, 38, 46, 52, 63],
    "maré":               [198, 236, 175, 168, 202, 172, 192, 222, 254],
    "complexo_do_alemao": [49, 74, 46, 55, 86, 76, 18, 16, 37],
    "jacarezinho":        [121, 140, 117, 126, 109, 128, 133, 146, 124],
}
# site -> (label, IBGE-2022 pop, pop_flag, AP)
SITES = {
    "rocinha":            ("Rocinha", 72021, "favela", "AP2.1"),
    "vidigal":            ("Vidigal", 15112, "favela", "AP2.1"),
    "maré":               ("Maré", 124832, "favela", "AP3.1"),
    "complexo_do_alemao": ("C. do Alemão", 54202, "favela⚠", "AP3.1"),
    "jacarezinho":        ("Jacarezinho", 37839, "2010-census", "AP3.1"),
}
# Alemão bairro TB is systematically under-ascribed (bairro split from Ramos → cases
# mis-attributed); keep it OFF the primary fit but report the ±Alemão sensitivity.
UNRELIABLE = {"complexo_do_alemao"}

AP_TB = {"AP2.1": (98.0, 108.9), "AP3.1": (103.9, 113.4), "AP4.0": (67.6, 72.5)}
AP_MEMBERS = {"AP2.1": ["rocinha", "vidigal"], "AP3.1": ["maré", "complexo_do_alemao"],
              "AP4.0": ["riodaspedras"]}


def sun_deficit(site: str, floor_h: float = 2.0) -> float:
    g = gpd.read_file(ROOT / f"outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg")
    col = "solar_hours_winter" if "solar_hours_winter" in g else "solar_hours"
    v = g[col].dropna()
    return float((v < floor_h).mean() * 100.0)


def incidence(site: str, years) -> float:
    """Mean annual TB incidence /100k over the given calendar years."""
    idx = [YEARS.index(y) for y in years]
    mean_cases = np.mean([TB_YEARLY[site][i] for i in idx])
    return mean_cases / SITES[site][1] * 1e5


def boot_ci(x, y, n=5000):
    """Bootstrap 95% CI of Spearman ρ by resampling the (x,y) points."""
    vals = []
    k = len(x)
    for _ in range(n):
        s = rng.integers(0, k, k)
        if len(set(x[s])) < 3 or len(set(y[s])) < 3:
            continue
        r, _ = spearmanr(x[s], y[s])
        if np.isfinite(r):
            vals.append(r)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)),
            float(np.median(vals)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sun = {s: sun_deficit(s) for s in SITES}
    inc9 = {s: incidence(s, YEARS) for s in SITES}          # 9-yr headline
    order = sorted(SITES, key=lambda s: -sun[s])

    print("=== TB incidence (9-yr mean 2015–23) vs winter sun-deficit ===")
    for s in order:
        print(f"  {SITES[s][0]:14s} sun<2h={sun[s]:5.1f}%  TB={inc9[s]:5.0f}/100k "
              f"(pop {SITES[s][1]:,} {SITES[s][2]}){'' if s not in UNRELIABLE else '  [excl.]'}")

    # ---- G4 robustness: ρ across window × ±Alemão × sun-threshold × leave-one-out ----
    specs, rob = [], []
    windows = {"2yr": [2022, 2023], "5yr": [2019, 2020, 2021, 2022, 2023], "9yr": YEARS}
    for wname, ws in windows.items():
        for excl in (True, False):
            sset = [s for s in SITES if not (excl and s in UNRELIABLE)]
            x = np.array([sun[s] for s in sset]); y = np.array([incidence(s, ws) for s in sset])
            r, pv = spearmanr(x, y)
            tag = f"{wname}/{'exclAlemão' if excl else 'inclAlemão'}"
            specs.append((tag, r, pv, len(sset))); rob.append(r)
    for floor in (1.0, 2.0, 3.0):  # sun-threshold sensitivity (excl Alemão, 9yr)
        sset = [s for s in SITES if s not in UNRELIABLE]
        x = np.array([sun_deficit(s, floor) for s in sset])
        y = np.array([incidence(s, YEARS) for s in sset])
        r, _ = spearmanr(x, y)
        specs.append((f"9yr/excl/<{floor:.0f}h", r, None, len(sset))); rob.append(r)
    # leave-one-out (9yr, incl Alemão, n=5)
    alls = list(SITES); loo = []
    for drop in alls:
        sset = [s for s in alls if s != drop]
        x = np.array([sun[s] for s in sset]); y = np.array([inc9[s] for s in sset])
        r, _ = spearmanr(x, y); loo.append((drop, r)); rob.append(r)

    frac_pos = np.mean([r > 0 for r in rob])

    # headline specs + bootstrap CI
    se = [s for s in SITES if s not in UNRELIABLE]
    xe = np.array([sun[s] for s in se]); ye = np.array([inc9[s] for s in se])
    rho_e, p_e = spearmanr(xe, ye)
    xa = np.array([sun[s] for s in SITES]); ya = np.array([inc9[s] for s in SITES])
    rho_a, p_a = spearmanr(xa, ya)
    ci_lo, ci_hi, ci_med = boot_ci(xa, ya)

    ap_pts = [(ap, np.mean([sun_deficit(s) for s in mem]), float(np.mean(AP_TB[ap])))
              for ap, mem in AP_MEMBERS.items()]
    rho_ap, _ = spearmanr([a[1] for a in ap_pts], [a[2] for a in ap_pts])

    print(f"\n=== SCORECARD ===")
    print(f"  G1 window .......... 9-yr (2015–23)  ✅")
    print(f"  G2 n ............... {len(SITES)} (incl Alemão) / {len(se)} (excl)  ✅")
    print(f"  headline ρ ......... excl Alemão n={len(se)}: {rho_e:+.2f} (p={p_e:.2f}) | "
          f"incl n={len(SITES)}: {rho_a:+.2f} (p={p_a:.2f})")
    print(f"  G3 bootstrap 95% CI (incl, n={len(SITES)}) ρ ∈ [{ci_lo:+.2f}, {ci_hi:+.2f}] (median {ci_med:+.2f})")
    print(f"  G4 robustness ...... ρ>0 in {frac_pos*100:.0f}% of {len(rob)} specs "
          f"(target ≥90%)  {'✅' if frac_pos >= 0.9 else '⚠'}")
    print(f"     spec range ρ ∈ [{min(rob):+.2f}, {max(rob):+.2f}];  "
          f"LOO ρ ∈ [{min(r for _, r in loo):+.2f}, {max(r for _, r in loo):+.2f}]")
    print(f"  AP washout ......... ρ={rho_ap:+.2f} (n=3 APs) — signal collapses when pooled")
    print(f"  G5 specificity ..... PENDING (leptospirosis placebo — verification cycle)")

    # ---- figure (multi-year) ----
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for s in SITES:
        rel = s not in UNRELIABLE
        c = "#b5651d" if rel else "#bbb"
        ax.scatter(sun[s], inc9[s], s=95, c=c, zorder=3, edgecolor="#333", linewidth=0.6)
        ax.annotate(SITES[s][0] + ("" if rel else " ⚠"), (sun[s], inc9[s]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9,
                    color="#333" if rel else "#999")
    b, a = np.polyfit(xe, ye, 1)
    xx = np.linspace(min(xa) - 3, max(xa) + 3, 50)
    ax.plot(xx, a + b * xx, "--", c="#b5651d", lw=1.2, alpha=0.7, zorder=2)
    ax.set_xlabel("Modelled winter sun-deficit  (% of street below WHO 2 h/day)")
    ax.set_ylabel("Tuberculosis incidence  (new cases /100k, 2015–23 mean)")
    ax.set_title("TB incidence vs winter sun-deficit — Rio favelas (bairro≈favela)",
                 fontsize=12, fontweight="bold")
    ax.text(0.02, 0.97,
            f"Spearman ρ = {rho_e:+.2f} (n={len(se)} excl. Alemão) · {rho_a:+.2f} (n={len(SITES)} incl.)\n"
            f"bootstrap 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}] · ρ>0 in {frac_pos*100:.0f}% of specs\n"
            f"ecological · hypothesis-generating · washes out at AP scale (ρ={rho_ap:+.2f})",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f2ec", ec="#e0d6c8"))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png = OUT / "tb_sun_deficit_screen.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\n  figure → {png.relative_to(ROOT)}")

    (OUT / "tb_sun_deficit_screen.json").write_text(json.dumps({
        "window_years": [YEARS[0], YEARS[-1]],
        "rows": [{"site": s, "label": SITES[s][0], "sun_pct_below_2h": round(sun[s], 1),
                  "tb_incidence_9yr": round(inc9[s], 1), "pop": SITES[s][1],
                  "reliable": s not in UNRELIABLE} for s in order],
        "headline": {"rho_excl_alemao": rho_e, "p_excl": p_e, "n_excl": len(se),
                     "rho_incl_alemao": rho_a, "p_incl": p_a, "n_incl": len(SITES),
                     "bootstrap_ci95": [ci_lo, ci_hi], "bootstrap_median": ci_med},
        "robustness": {"frac_specs_positive": frac_pos, "n_specs": len(rob),
                       "rho_range": [min(rob), max(rob)],
                       "leave_one_out": {d: r for d, r in loo}},
        "ap_washout_rho": rho_ap,
        "specificity_placebo": "pending",
        "framing": "ecological, hypothesis-generating; NOT causal; support-dependent",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
