# Why the P1 flagship stops at the adequacy surface — the ecological-linkage attempt

> **Purpose (T0 of the P1-council handoff).** A citable block for the brisaverse P1
> Methods/SI. It converts the ecological-fallacy objection into a *declared design choice*:
> we attempted the direct exposure→outcome linkage, and its honest properties are precisely
> why the flagship reports a built-environment **adequacy surface**, not a health-outcome model.
> Every number here already passed the health track's adversarial-verify gate
> (`docs/planetary_health_plan.md`, G6). External citations are flagged for human re-check.

## The attempt

We paired the modelled winter direct-sun deficit (share of favela fabric below the
WHO ≥2 h/day floor) against **real tuberculosis incidence** — SMS-Rio SINAN notifications by
bairro de residência, 9-year mean (2015–2023), over IBGE-2022 populations — for the five Rio
favelas where a bairro is essentially a single favela (Rocinha, Jacarezinho, Vidigal, Complexo
do Alemão, Maré). TB was chosen because it shares the exact etiological axes we model —
crowding, low ventilation, low sunlight — and because vitamin-D deficiency (the sun→health
mechanism) is a replicated TB susceptibility factor.

## What we found (and why it does not become a flagship outcome)

| Property | Result | Consequence |
|---|---|---|
| Direction | Spearman **ρ = +0.80** (n=5) — sun-starved favelas carry more TB | consistent with the mechanism |
| Significance | exact two-tailed permutation **p ≈ 0.13** — **not significant** | n=5 cannot carry an outcome claim |
| Spatial support | **reverses to ρ ≈ −0.50** when pooled to Área de Planejamento | a textbook MAUP / change-of-support instability |
| Specificity | powered dengue placebo (4,380 cases) **ρ = +0.10** vs TB +0.80 | rules out *generic deprivation tracks everything*… |
| Confounding | sun-deficit is collinear with density/crowding/poverty | …but does **not** isolate a sun mechanism from indoor crowding |

**One-line finding:** the ecological linkage is *direction-consistent but underpowered
(n=5, n.s.), scale-sensitive (MAUP sign-flip), and confound-limited (crowding-inseparable)* —
so it is a hypothesis-generating probe, not an outcome model.

## Why this justifies the firewall

An n=5, non-significant, scale-unstable disease×geometry correlation is (a) a reviewer-kill at
*Lancet Planetary Health* and (b) exactly the artifact the P1 firewall exists to prevent — a
map that reads as "this fabric causes disease" and can be weaponized against residents.
Therefore the flagship makes **no health-outcome claim**; it reports where the built
environment produces unequal, WHO-referenced **exposure** (an *adequacy surface*), and the
health consequences are stated as literature mechanisms, not findings. The linkage above is the
empirical evidence that this is the honest ceiling, not a hedge of convenience — we ran it, and
it told us to stop at exposure.

Two hard boundaries carried from the health track and inherited by P1:
- **No per-cell disease surface, ever.** Ecological (bairro/setor) only. A per-cell sun→TB map
  is the single most weaponizable artifact the line could ship.
- **Ecological fallacy is declared, not discovered:** a bairro rate is not an individual risk;
  a genuine linkage needs individual-level data behind CEP-CONEP ethics (parked).

## The mechanism (literature only — VERIFY bylines before print)

Vitamin D is the plausible *why*, not a measured endpoint: winter sun-deficit lowers UVB →
lower cutaneous 25(OH)D. Rio priors — Pró-Saúde cohort (**+0.49 nmol/L per unit sun-exposure**,
**+20 nmol/L** summer vs winter; *Cad. Saúde Pública* 2022;38(1):e00287820) — and the TB bridge
(vitamin-D deficiency ↔ active TB, pooled **OR 3.23**; Cureus 2021;13(9):e17883). **These
citations must be re-checked against PubMed/PMC/SciELO before publishing** (the line has a
recurring agent-fabrication hazard; flag, do not trust).

---
*Provenance: `scripts/health/tb_sun_deficit_screen.py` → `outputs/comparative/health/`.
Numbers current as of health-track loop Cycle 6 (commit d70f32d).*
