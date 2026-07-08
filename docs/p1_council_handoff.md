# P1-council handoff → planetary-health track task spec

> **Source.** The brisaverse P1 flagship ran a 6-seat decision council (2026-07-08) on the
> PI (Fabio) review of the Lancet Planetary Health draft. Several of its conclusions bear
> directly on this health-linkage track. This doc translates them into concrete, ranked tasks
> for the orchestrator/coding-analysis agent. Fold these into `planetary_health_plan.md`'s
> Priorities table on the next cycle; keep the scorecard/adversarial-verify discipline.

## The one strategic decision this encodes

The TB×sun-deficit ecological screen (ρ=+0.80, n=5, p≈0.13 **n.s.**, MAUP washout, crowding-
confounded) is **NOT going into the P1 flagship as a health outcome** — an n=5 non-significant
disease×geometry correlation is both a reviewer-kill at LPH and a re-arming of the exact
weaponization the P1 firewall exists to prevent. Instead:

1. **To P1 it contributes ONE thing:** the empirical justification for P1's outcome-free
   design (see T0). "We attempted the ecological linkage; it is direction-consistent but
   underpowered, scale-sensitive, and confound-limited — hence an adequacy surface, not an
   outcome model." That converts the ecological-fallacy objection into a declared design choice.
2. **The screen itself is the seed of a SEPARATE health-linkage output** (companion / this
   track), whose binding constraints are **n** (data-gated DTM onboarding — user) and
   **individual-level ethics** (CEP-CONEP). Maximise rigor-at-fixed-n; do not smuggle it into
   the flagship.

### Guardrails inherited from the P1 council (non-negotiable)
- **No per-cell disease map, ever.** Ecological (bairro/setor) only, hypothesis-generating only.
  A per-cell sun→TB surface is the single most weaponizable artifact the line could ship.
- **CFD does not gate anything here.** The ventilation exposure for this track is the
  *morphometric* proxy (skimming-regime / dissolved λf), which is available now — NOT the
  OpenFOAM τ field (one partial pilot, ABL-fail, synthetic per-site trees; keep it out).
- **External citations need human verification before publishing** (Pró-Saúde byline, vitamin-D↔
  TB OR 3.23, all PMC IDs) — the line has a recurring agent-fabrication hazard; re-check vs
  PubMed/PMC/SciELO. Flag, don't trust.

---

## Ranked tasks

### T0 — Firewall-justification handback to P1 *(HIGH · one-shot · unblocks P1 prose)*
Produce a tight, citable markdown block summarising the honest ecological result — ρ=+0.80,
n=5, exact permutation p≈0.13 (n.s.), AP-scale sign reversal (−0.50), dengue-placebo
specificity (+0.10) but crowding-inseparable — framed as *the empirical reason P1 stops at the
adequacy surface*. This is the "we ran the linkage and it is why we make no outcome claim"
paragraph the LPH editor seat asked for. Deliver to `outputs/comparative/health/` + a copy the
brisaverse P1 Methods/SI can drop in verbatim.
- **Serves:** P1 D1 (health framing) + the ecological-fallacy pre-emption.

### T1 — Cross-repo exposure reconciliation *(HIGH · consistency)*
The screen's sun-deficit is **street-observer** `solar_hours_winter<2h` (Maré 27.8%); P1
canonical (`brisaverse/shared/facts/solar_canonical.json`) is **built-cell** %<2h (Maré 35%,
Rocinha 74, Vidigal 55, Alemão 42, pooled 46). Rocinha/Vidigal/Alemão agree within ~2pt but
**Maré diverges 7pt** (open fabric → observer vs cell denominators diverge most).
- Re-run the screen on the **canonical built-cell** exposure so both repos report ONE number
  (brisaverse rule: facts shared, not copied), OR document the observer-vs-cell delta explicitly
  and pick the canonical one deliberately.
- Report whether ρ changes under the canonical exposure (rank-robustness expected, but verify).
- **Serves:** P1 D2 (the unit is *built cells*, not surfaces/observers) + cross-line consistency.

### T2 — Crowding/deprivation covariate *(HIGH · the specificity frontier)*
The honest ceiling is "sun-deficit cannot be separated from indoor crowding." Add IBGE-2022
favela-flagged-setor crowding (persons/household, density; dataset #3 already queued) and report
the **partial rank-association ρ(sun, TB | crowding)**. Does sun-deficit retain *any* signal
beyond crowding? This is the single highest-value robustness step and it is open data, runnable
now. Report honestly either way (a null here is a real, publishable finding about the limit).
- **Serves:** the health-linkage output's credibility; directly answers the council's "generic
  deprivation" alternative.

### T3 — Terrain-aspect vs morphology decomposition of the exposure *(MEDIUM · aligns to P1's strongest result)*
P1's headline novel finding: winter-sun deficit is driven by **terrain aspect (south-facing
slope), not SVF/openness**, and the within-favela south-vs-north equal-slope contrast (1.4–2.9 h)
is a terrain- and class-controlled natural experiment. Decompose each favela's sun-deficit into
**terrain-driven (immutable)** vs **morphology-driven (in-situ fixable)** components using the P1
slope×northness machinery, and prepare the per-site decomposed exposures so the terrain-vs-
morphology health contrast is ready when n grows. (n=5 is too small to test now — build the
design + the exposure split.)
- **Serves:** P1 D3-C (terrain/morphology decomposition) + gives the health track a
  confound-controlled exposure instead of a raw favela-level %.

### T4 — Compound (sun × morphometric-ventilation) exposure *(MEDIUM · aligns to P1 4-state taxonomy)*
The screen is sun-only. Add the **morphometric ventilation index** (skimming-regime membership /
dissolved λf > 0.65 — available now, NOT the CFD) as a second exposure and test the P1 4-state
**compound-deficit** against TB. Does compound (sun+ventilation) deficit track TB better than sun
alone? This tests P1's compound taxonomy against a health endpoint, ecologically, and gives the
council's "respiratory mechanism lives on ventilation" claim an empirical touchpoint — without
touching the gated CFD.
- **Serves:** P1 D1 (respiratory→ventilation) + validates the compound taxonomy's health relevance.

### T5 — Formal power analysis + n-onboarding target list *(MEDIUM · turns the ceiling into a user action)*
The binding constraint is n≈8 (data-gated). Produce (a) a **permutation power curve**: what n
detects ρ≈0.8 at p<0.05? and (b) a **ranked target list** of favela-bairros that have BOTH
TabNet TB-by-bairro AND feasible DTM onboarding, so the user knows exactly which DTMs to clip to
move the needle. Converts "n is gated" into a concrete, prioritised onboarding worklist.
- **Serves:** unblocks the whole track's statistical power; hands the user a decision.

### T6 — CEP-CONEP protocol draft *(LOW · parallel · user-gated)*
Draft the address-level SINAN-TB / SIM microdata protocol (datasets #2/#7) scoped to the 5 sites'
setores — the only path to a publishable *individual-level* linkage (the health-linkage companion's
long pole). Agent drafts; user files. Keep parked but draft-ready.
- **Serves:** the health-linkage companion's eventual publishability.

---

## Sequencing
Ship **T0 + T1 + T2** first (all runnable now, no new data, and they respectively: unblock P1
prose, fix the cross-repo number, and attack the confound that most threatens the claim). Then
**T3/T4** (align the exposure to P1's strongest science). **T5** whenever, to hand the user the
onboarding list. **T6** parallel/parked.

Every new health number still passes the existing G6 adversarial-verify gate before it reaches
`health.html`, and the exposure-not-outcome + ecological-fallacy hedges stay load-bearing.
