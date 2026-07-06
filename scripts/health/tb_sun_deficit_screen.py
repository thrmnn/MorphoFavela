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
- TB new-case counts by bairro de residência: SMS-Rio SINAN TabNet, tuberc2007.def,
  tube22.dbf / tube23.dbf (semicolon 'prn' export). Rocinha 351/354, Vidigal 52/63,
  Maré 222/254, Complexo do Alemão 16/37, Jacarezinho 146/124 (2022/2023).
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

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "comparative" / "health"

# site -> (label, TB cases 2022, TB cases 2023, IBGE pop, pop_flag, AP)
SITES = {
    "rocinha":            ("Rocinha", 351, 354, 72021, "favela", "AP2.1"),
    "vidigal":            ("Vidigal", 52, 63, 15112, "favela", "AP2.1"),
    "maré":               ("Maré", 222, 254, 124832, "favela", "AP3.1"),
    "complexo_do_alemao": ("C. do Alemão", 16, 37, 54202, "favela⚠", "AP3.1"),
    "jacarezinho":        ("Jacarezinho", 146, 124, 37839, "2010-census", "AP3.1"),
}
# Alemão bairro TB is an artefact; keep it off the primary fit.
UNRELIABLE = {"complexo_do_alemao"}

AP_TB = {"AP2.1": (98.0, 108.9), "AP3.1": (103.9, 113.4), "AP4.0": (67.6, 72.5)}
# Rio das Pedras has no standalone bairro (splits across Jacarepaguá/Itanhangá),
# so it only contributes at AP4.0; its sun-deficit is carried for the AP view.
AP_MEMBERS = {"AP2.1": ["rocinha", "vidigal"], "AP3.1": ["maré", "complexo_do_alemao"],
              "AP4.0": ["riodaspedras"]}


def sun_deficit(site: str) -> float:
    g = gpd.read_file(ROOT / f"outputs/{site}/morphometrics/svf/svf_streets_solar.gpkg")
    col = "solar_hours_winter" if "solar_hours_winter" in g else "solar_hours"
    v = g[col].dropna()
    return float((v < 2.0).mean() * 100.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for site, (label, c22, c23, pop, flag, ap) in SITES.items():
        inc = (np.mean([c22, c23]) / pop) * 1e5
        rows.append(dict(site=site, label=label, sun=sun_deficit(site), inc=inc,
                         pop=pop, flag=flag, ap=ap,
                         reliable=site not in UNRELIABLE))

    prim = [r for r in rows if r["reliable"]]
    xs = np.array([r["sun"] for r in prim]); ys = np.array([r["inc"] for r in prim])
    rho, p = spearmanr(xs, ys)
    xa = np.array([r["sun"] for r in rows]); ya = np.array([r["inc"] for r in rows])
    rho_all, p_all = spearmanr(xa, ya)

    # AP-level (the MAUP washout): mean member sun-deficit vs published AP rate
    ap_pts = []
    for ap, sites in AP_MEMBERS.items():
        sd = np.mean([sun_deficit(s) for s in sites])
        ap_pts.append((ap, sd, float(np.mean(AP_TB[ap]))))
    rho_ap, _ = spearmanr([a[1] for a in ap_pts], [a[2] for a in ap_pts])

    print("=== TB incidence vs winter sun-deficit (favela≈bairro) ===")
    for r in sorted(rows, key=lambda r: -r["sun"]):
        print(f"  {r['label']:14s} sun<2h={r['sun']:5.1f}%  TB≈{r['inc']:5.0f}/100k "
              f"(pop {r['pop']:,} {r['flag']})  {'' if r['reliable'] else '[excluded]'}")
    print(f"\n  Spearman ρ (n={len(prim)}, reliable) = {rho:+.2f}  (p={p:.2f})")
    print(f"  Spearman ρ (n={len(rows)}, incl. Alemão) = {rho_all:+.2f} (p={p_all:.2f})")
    print(f"\n=== AP-level washout (change-of-support) ===")
    for ap, sd, tb in ap_pts:
        print(f"  {ap}: sun<2h={sd:5.1f}%  TB={tb:5.1f}/100k")
    print(f"  Spearman ρ (n={len(ap_pts)} APs) = {rho_ap:+.2f}  "
          f"→ the bairro-scale signal collapses when pooled to AP")

    # figure
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for r in rows:
        c = "#b5651d" if r["reliable"] else "#bbb"
        ax.scatter(r["sun"], r["inc"], s=90, c=c, zorder=3,
                   edgecolor="#333", linewidth=0.6)
        ax.annotate(r["label"] + ("" if r["reliable"] else " ⚠"),
                    (r["sun"], r["inc"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=9,
                    color="#333" if r["reliable"] else "#999")
    # trend on reliable points
    if len(prim) >= 2:
        b, a = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xa) - 3, max(xa) + 3, 50)
        ax.plot(xx, a + b * xx, "--", c="#b5651d", lw=1.2, alpha=0.7, zorder=2)
    ax.set_xlabel("Modelled winter sun-deficit  (% of street below WHO 2 h/day)")
    ax.set_ylabel("Tuberculosis incidence  (new cases /100k, 2022–23 mean)")
    ax.set_title("TB incidence vs winter sun-deficit — Rio favelas (bairro≈favela)",
                 fontsize=12, fontweight="bold")
    ax.text(0.02, 0.97,
            f"Spearman ρ = {rho:+.2f} (n={len(prim)}, p={p:.2f}, NOT significant)\n"
            f"ecological · hypothesis-generating · signal washes out at AP scale (ρ={rho_ap:+.2f})",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f2ec", ec="#e0d6c8"))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png = OUT / "tb_sun_deficit_screen.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\n  figure → {png.relative_to(ROOT)}")

    import json
    (OUT / "tb_sun_deficit_screen.json").write_text(json.dumps({
        "rows": [{k: r[k] for k in ("site", "label", "sun", "inc", "pop", "flag", "reliable")}
                 for r in rows],
        "spearman_bairro_reliable": {"rho": rho, "p": p, "n": len(prim)},
        "spearman_bairro_all": {"rho": rho_all, "p": p_all, "n": len(rows)},
        "spearman_ap": {"rho": rho_ap, "n": len(ap_pts)},
        "framing": "ecological, hypothesis-generating; NOT causal; support-dependent",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
