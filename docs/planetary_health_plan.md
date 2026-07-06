# Planetary-health track — living orchestration plan

> **Role.** This plan is driven by a high-level **orchestrator agent** running a team of
> subagents through **dynamic workflows** (fan-out → adversarial/verify → synthesize).
> It is a *living* document: **each cycle appends the single most important item** to the
> "Cycle log" at the foot, and re-ranks the tracks. The hub section it governs
> (`outputs/_hub/health.html`) is regenerated, never hand-edited.

*Companion to [[project-health-dashboard]]. The shipped exposure section is the
foundation; this plan is about what we build on top of it and, above all, whether we can
anchor it to **real Rio de Janeiro health data**.*

---

## Priorities — redefined 2026-07-05 (with the user)

| # | Track | Status | Note |
|---|-------|--------|------|
| **P0** | **TB×sun-deficit open ecological screen** — first REAL health number | **DONE (Cycle 3)** | ρ=+0.80 at favela scale (n=4, p=0.20, NOT sig); washes out at AP scale (ρ=−0.50). `scripts/health/tb_sun_deficit_screen.py`. |
| **P0-mech** | **Vitamin-D mechanism narrative** (Pró-Saúde + lab priors) | **DONE (Cycle 3)** | sourced: Pró-Saúde β=+0.49 nmol/L per sun-unit, +20.1 summer swing; TB bridge OR 3.23. Ready to drop into health.html. |
| **P0-next** | **Power the screen: n=4 is the bottleneck** | **NEXT** | model sun-deficit for more favela-bairros (n→20+) so the ecological screen has power; setor/point join is ethics-gated (parked) |
| **P0-fabric** | IBGE favela-flagged setores GeoPackage (#3) — the join key | queued | needed for setor down-weighting + covariate adjustment |
| P1 | Surface the TB probe + vitamin-D mechanism on `health.html` | queued | new honesty tier: "outcome probe (ecological)", below Grade A |
| P1 | **CEP-CONEP protocol** for address-level SINAN-TB/SIM microdata (#2/#7) | queued (user-parked) | the publishable point-join; long pole |
| P1 | Equity **inequality-strip** figure (Gini shown, not just cited) | queued | HE2 |
| P2 | Heat pathway (**Grade D**) → real thermal proxy; TR cross-links | queued | HE3/HE4 |
| **PARKED** | Airflow / CFD ventilation leg C→B (HE1) | **WAIT** | out of scope — gated on MAR-P07; do not chase |

**Reading of the redefinition (updated post-scout):** stop waiting on CFD; find *real
health signal for Rio* instead. The honest result: **vitamin D can't be validated at favela
scale**, so it becomes the mechanism narrative (real Rio priors) and **georeferenced TB is
the tested endpoint** — same crowding/ventilation/sunlight etiology, and favela-resolvable.
The open bairro-level TB screen is runnable now; the address-level point join is the
publishable upgrade gated on ethics approval.

---

## The foundation (already shipped — do not redo)

- Exposure section `health.html`: 4 pathways, A–D evidence grades, exposure-not-outcome
  disclaimer, compound-deprivation table bound to `cross_site_stats.json`.
- Grade-A surface = winter direct-sun deficit vs WHO-2h (AUC 0.90, Ladybug-cross-checked).
- Honest limit today: **zero health-outcome data**. This plan attacks exactly that.

---

## Council of experts — who set these priorities

The priorities above are not ad-hoc: they are the output of successive **expert panels**,
each run as a subagent team and each grounded only in what the repo actually measures.
Three have convened.

**Panel 1 — Lancet Planetary Health panel** *(built `health.html`; Cycle 0).* Four
disciplinary advocates argued distinct pathways against one skeptic:
- **Urban-heat / climatology** — the SVF double-edge (daytime shade vs nocturnal trapping).
- **Respiratory / infectious-disease epidemiology** — ventilation + germicidal sunlight → TB / respiratory.
- **Healthy-housing / solar deprivation** — winter-sun deficit → damp, mould, vitamin D, mood (the strongest).
- **Health equity / environmental justice** — distribution of exposure (Gini), transferable risk.
- **Methods-editor skeptic (chair)** — fixed the **A–D evidence rubric**, banned causal verbs
  and any invented incidence, and wrote the mandatory **exposure-not-outcome disclaimer**.

Output → the four graded pathways on the section (solar = Grade A anchor; equity B/D;
airborne B/C; heat D). *This is why the section is defensible: no pathway claims more than
its grade.*

**Panel 2 — RJ health-data scout** *(set the P0 pivot; Cycles 1–2).* Five scouts (SINAN/TB,
vitamin-D, mortality/heat, respiratory/damp, geo-join+access) + a high-effort synthesis;
ranked 48 datasets by relevance × joinability × access. Output → the **vitamin-D → TB pivot**
and the ranked dataset table below.

**Panel 3 — Cycle-3 execution pair.** One agent retrieved real TB-by-bairro (SMS-Rio TabNet);
one built the vitamin-D mechanism from real Rio priors. Output → the first real health number
+ the sourced mechanism.

**How the council sets priorities.** A track is P0 only if it clears the skeptic's grade *and*
the scout's join test — it must (a) map to a real exposure surface we own, (b) resolve to
favela fabric, and (c) be accessible. That triple is why **TB is P0** (open, favela-resolvable,
matches our surfaces), why **vitamin-D-as-outcome is not** (no favela-scale serum data → it is
mechanism only), and why **airflow is parked** (CFD synthetic + gated). The Priorities table at
the top *is* the council's current ranking; each Cycle-log entry records how it moved.

---

## Track P0 — real RJ health data (scouted 2026-07-05)

> From `rj-health-data-scout` (6 agents, 48 candidate datasets, 0 errors; fan-out over
> SINAN/TB, vitamin-D, mortality/heat, respiratory/damp + the geo-join/access layer →
> high-effort synthesis). Ranked by **relevance × joinability-to-favela-fabric × access**.

**The join question (decides everything):** finest → coarsest unit that can bind a health
number to a favela — CEP-geocoded **point** → IBGE **setor censitário** (finest OPEN unit
that still *tags* favela) → IBGE **aglomerado subnormal** polygon → Data.Rio **bairro** →
**AP** → município. Public DATASUS strips address to município (all-of-Rio, ~6.7M) — so the
open tier is bairro-level *ecological* at best; true favela resolution needs CEP-CONEP ethics.

| # | Dataset | Pathway | Unit | Join | Access | First step |
|---|---------|---------|------|------|--------|-----------|
| **1** | **SMS-Rio municipal TabNet — SINAN-Tuberculose** (tabnet.rio.rj.gov.br) | TB | **bairro** de residência | partial (down-weight to favela setores) | **open** | pull TB incidence by bairro for the 5 sites + comparison bairros |
| 2 | SINAN-TB **address/CEP microdata** (SMS-Rio Vigilância) | TB | **CEP/point** | yes (point-in-polygon) | **ethics** (CEP-CONEP + DUA) | file the protocol now, scoped to the 5 sites' setores |
| **3** | **IBGE Favelas e Comunidades Urbanas polygons + favela-flagged setores 2022** | *join fabric* | setor/polygon | **yes — the canonical key** | **open** | download RJ setor GeoPackage, filter favela setores, clip our rasters |
| 4 | Published setor-level TB spatial analysis (PMC4544397) | TB | setor | yes (concept) | read-open | template + names **Rocinha/Vidigal** hotspots → sanity-check our surface |
| 5 | SIH-SUS respiratory AIH (J-codes) via PCDaS/Fiocruz | respiratory | CEP→setor | partial | request (DUA) | register PCDaS, pull RJ J00–J99, quantify favela geocoding loss |
| 6 | SISAB / e-SUS APS — **Clínica da Família catchments** | respiratory | facility micro-area ≈ favela | partial (no geocoding needed) | open (aggregate) | map the CdF units serving each site → asthma/DPOC attendance |
| 7 | SIM identified microdata (address/CEP) | mortality | CEP/point | yes | ethics | bundle respiratory + heat causes into the #2 protocol |
| 8 | SMS-Rio TabNet — SIM/SIH by bairro | mortality/resp | bairro/AP | partial | open | tabulate respiratory-cause mortality by bairro alongside #1 |

### Vitamin-D verdict (the favored track — honest answer)

**A real vitamin-D *validation* at favela resolution is NOT feasible** — structural gap, not
a search failure: (1) serum 25(OH)D at favela/setor scale does not exist; (2) the ICD proxy
E55 in SIH/SIA is município-only and too rare to spatialize; (3) PNS 2013/2019 deliberately
excluded vitamin D, so no national serum-D surface exists either. **So vitamin D moves from
"outcome to validate" → "mechanism to narrate," anchored on real Rio priors:**

- **Pró-Saúde Study** (UERJ cohort; *Cad. Saúde Pública* 2022;38(1):e00287820, n=491) — the
  only Rio study quantifying the **sun-exposure-index → 25(OH)D** coefficient (β≈0.49/unit)
  and season effect (≈20 nmol/L summer). Non-favela (UERJ civil servants), no geocoding →
  anchors the mechanism, cannot be spatially joined.
- **2018 Rio clinical-lab cross-section** (n≈24,074; PMC8009065) — the strong seasonal 25(OH)D
  swing that justifies our winter focus; private-lab selection skews from favela residents.

**Therefore the tested downstream endpoint is georeferenced tuberculosis** — vitamin-D
deficiency is an established TB susceptibility factor, TB shares our exact etiology (crowding
+ low ventilation + low sunlight), and TB *is* favela-resolvable. Vitamin D is the *why*; TB
is the *measurable*.

### Recommended first experiment (open, zero-approval, runnable now)

Bairro-level **ecological screen**: SMS-Rio TabNet **TB incidence by bairro** → down-weight
each bairro to its **favela fraction** via IBGE 2022 favela-flagged setores (#3) → regress on
our **winter sun-deficit surface** (WHO-2h, AUC 0.90; SVF + ventilation index as secondary),
negative-binomial or Spearman, across the 5 sites' bairros + a comparison set. **Reported as
hypothesis screening, never causal.** Produces the project's first REAL health number,
cross-checks the published Rocinha/Vidigal setor hotspots, and defines exactly the CEP-CONEP
request needed to upgrade to the address-level point join.

### Hard caveats (gate every claim)

Ecological fallacy (bairro ≠ favela) · MAUP / change-of-support (our own §10.9 skimming
reversal is the in-house proof) · privacy suppression (public data stops at município) ·
**geocoding bias inside favelas** (~11% record loss, ~27% address correction — undercounts
the densest fabric we model) · confounding (TB co-varies with income/crowding/HIV) · temporal
mismatch (static exposure vs dated notifications) · **vitamin-D unfalsifiable at favela scale**.

### RESULT — first real health number (Cycle 3, 2026-07-06)

`scripts/health/tb_sun_deficit_screen.py` → `outputs/comparative/health/tb_sun_deficit_screen.{png,json}`.
Real TB new-case counts (SMS-Rio SINAN TabNet, 2022–23) ÷ IBGE-2022 population, paired with
our modelled winter sun-deficit, for the favela≈bairro sites:

| favela | sun-deficit (% <2 h) | TB incidence /100k (22–23) | note |
|---|---|---|---|
| Rocinha | 72.4 | ≈489 | 447/100k in 2004–06 (PMC4544397) → persisted + intensified |
| Jacarezinho | 70.2 | ≈357 | historic TB epicentre; independent calibration favela |
| Vidigal | 55.5 | ≈380 | small pop → noisy rate |
| Maré | 27.8 | ≈191 | most open fabric, lowest TB |
| C. do Alemão | 44.1 | ≈49 ⚠ | **excluded** — bairro count is a notification artefact (16→37) |

- **Bairro≈favela: Spearman ρ = +0.80** (n=4 reliable, p=0.20; n=5 incl. Alemão, p=0.10) —
  a strong positive rank association, **NOT statistically significant** (tiny n). The
  sun-starved favelas carry 2–3× the TB of the open Maré.
- **AP-level: ρ = −0.50** — the signal **washes out (even reverses) when pooled to Área de
  Planejamento**, because AP mixes favela + asfalto and pools low-deficit Maré with others.
  A live, real-data instance of the change-of-support caveat (mirrors our §10.9).
- **Framing:** ecological, hypothesis-generating, consistent with the vitamin-D/germicidal
  mechanism — **not** a causal or validated claim. n is the bottleneck.

### Vitamin-D mechanism (Cycle 3 — sourced, ready for health.html)

The *why* behind the TB probe (vitamin D is not measurable at favela scale, so it is
mechanism, not outcome):

- **Pró-Saúde** (UERJ cohort, n=491; *Cad. Saúde Pública* 2022;38(1):e00287820):
  serum 25(OH)D **+0.49 nmol/L per unit sun-exposure index** (CI 0.22–0.75), **+20.14 nmol/L
  summer vs winter** (CI 14.38–25.90); 55% deficient. The direct Rio sun→25(OH)D quantum.
- **Leão et al.** *Clinics* 2021;76:e2571 (n=24,074, summer-only): 50.6–53.2% of seniors
  <30 ng/mL even at the seasonal peak → the winter trough is worse.
- **Bridge to TB:** vitamin-D deficiency ↔ active TB, pooled **OR 3.23** (CI 1.91–5.45;
  Cureus 2021;13(9):e17883, *external*). WHO Housing and Health Guidelines 2018 (*external*).
- *Caveat:* both Rio cohorts are non-favela and ungeocoded → mechanism anchor, not a joinable
  layer. One author byline (Pró-Saúde) to verify against SciELO before publishing.

---

## Orchestration protocol (how each cycle runs)

1. **Pick the one most important open item** (top of the ranked tracks).
2. **Fan out** a subagent team via a dynamic workflow sized to the task (scout / verify /
   synthesize; adversarial verification for any claim that would touch the public section).
3. **Synthesize** into a concrete artifact or a ranked decision.
4. **Append** the item + outcome to the Cycle log; re-rank tracks.
5. **Never** let an unverified health claim reach `health.html`; the exposure-not-outcome
   disclaimer and the A–D grades are load-bearing.

## Guardrails (non-negotiable)

- **Exposure ≠ outcome** until a real, spatially-joined dataset says otherwise — and even
  then, report association with the ecological-fallacy / MAUP / privacy-suppression
  caveats attached.
- **Ethics/access is a gate, not an afterthought** — individual-level DATASUS linkage may
  need CEP / Plataforma Brasil approval; open aggregate data comes first.
- **Generators only** for the hub; test-staged commits; airflow stays parked.

---

## Cycle log (append-only — newest first)

### Cycle 3 — 2026-07-06 · first REAL health number + sourced vitamin-D mechanism
- Ran both user-chosen tracks in parallel via subagents: pulled real TB-by-bairro from
  SMS-Rio TabNet (working scrape recipe) + built the vitamin-D mechanism from real Rio priors.
- Assembled the **TB × sun-deficit ecological screen** myself over n=8 favela sun-deficit
  values (extended to include Jacarezinho, the TB epicentre): **ρ=+0.80 at favela scale
  (n=4, NOT sig), washes out at AP scale (ρ=−0.50)**. First real health number in the project.
- **Most important item added:** **the finding is real but underpowered (n=4) and confounded**
  — the binding constraint is now *sample size*, not data access. Since the setor/point join
  is ethics-gated (user-parked), the next lever is **modelling sun-deficit for more Rio
  favela-bairros** to power the ecological screen. Also logged: the AP-scale washout is a
  publishable methodological point (change-of-support), not a failure.

### Cycle 2 — 2026-07-05 · scout returned → the TB pivot
- `rj-health-data-scout` returned: 6 agents, 48 datasets, 0 errors. Ranked table + vitamin-D
  verdict + first experiment + caveats folded into Track P0 above.
- **Most important item added:** **the vitamin-D → TB pivot.** A real vitamin-D validation is
  structurally impossible at favela scale (no serum 25(OH)D there, E55 município-only, PNS
  excluded it). So vitamin D becomes the *mechanism* (Pró-Saúde β=0.49 prior) and
  **georeferenced TB becomes the tested endpoint** — same etiology, and favela-resolvable via
  IBGE favela-flagged setores. This is the through-line that makes the whole health track
  falsifiable instead of proxy-only.
- Re-ranked: P0 is now the *open TB×sun-deficit ecological screen* + its IBGE favela-setor
  join key; CEP-CONEP microdata protocol is the P1 long pole to start in parallel.

### Cycle 1 — 2026-07-05 · redefine priorities + launch the RJ health-data scout
- Redefined priorities with the user: **airflow parked**, **real RJ health data = P0**,
  **vitamin D favored**. Created this living plan.
- Launched `rj-health-data-scout` (5-angle fan-out + synthesis) to find outcome data
  joinable to the exposure surfaces.
- **Most important item added:** the *join question* — no health track is viable unless it
  resolves to favela fabric (CEP / setor / IBGE aglomerado subnormal), so joinability is
  the top ranking axis, above raw relevance. _(Track P0 table lands when the scout returns.)_
