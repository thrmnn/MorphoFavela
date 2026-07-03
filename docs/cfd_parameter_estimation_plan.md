<!-- /autoplan restore point: /home/theo/.gstack/projects/thrmnn-MorphoFavela/main-autoplan-restore-20260703-025629.md -->
# CFD parameter-estimation plan — extending the morphometric → CFD hand-off

> **Status: REVIEWED + DECIDED — validate-first (2026-07-03).** A 3-lens `/autoplan`
> review (CEO/Eng/DX, see the review section at the foot of this doc) challenged the
> original "build F/W/I now" direction; the user chose **validate-first**. Chosen path:
> **(1) ship the one genuinely-morphometric piece — NaN-safe floored z0 + flag +
> provenance on the hand-off (DONE, commit `9a509d1`); (2) request the single MAR-P07×N
> real-CFD pilot as the R-C anchor (DONE — `src/cfd_integration/README.md` §Pilot
> request). Tracks F (fetch z0), I (inlet fields), W's `ks`/two-zone, P, and D are
> DEFERRED — gated on that pilot recalibrating Kanda.** Their hardened build specs are
> retained below and in the review section so they are execute-ready the moment the
> pilot greenlights. Local-compute only; no CFD execution here.
>
> **Why not build them now:** F/W/I are all propagation of the same Kanda estimator the
> roughness council flagged as ~1.5 orders uncertain and invalid in 53–75 % of cells.
> One real return decides whether they are worth building. See review Finding 1.

## Why this exists

The roughness track shipped the patch-scale morphometric `z0(θ)/zd(θ)` and the
`patch_roughness.csv` hand-off. But the CFD side needs more than a roughness length to
stand up a physically-consistent case, and the coupling contract already names two gaps
we left for the CFD agent to improvise:

1. **The inlet roughness is sourced from the wrong place.** The contract's council
   refinement #1 is explicit: `z0_inlet(θ)` must come from the **upwind fetch**, not the
   patch cell — conflating them double-counts the patch's own (already-meshed) buildings.
   The contract calls this "the single biggest coupling risk." Today `build_patch_roughness.py`
   only emits patch z0. **No upstream-fetch column family exists.**
2. **Wall treatment is undefined at the boundary MorphoFavela owns.** For λp>0.5 skimming
   patches the morphometric z0 collapses to ~0 (invalid as a log denominator). The contract
   mandates a floor + an equivalent sand-grain `ks` for wall functions — but we emit only a
   boolean flag, forcing the CFD agent to invent the floor, the `ks`, and the two-zone split.

The rest of what CFD needs (inlet turbulence fields, blockage, domain extents) is either a
one-line derivation from roughness we already have, or already solved
(`rectangular_domain_v1.json`). This plan closes the two real gaps and productionizes the
cheap derivations, so the hand-off is a complete inlet+wall specification, not a roughness
number the CFD side has to wrap physics around.

## What already exists (do not rebuild)

| Artifact | Covers |
|----------|--------|
| `patch_roughness.csv` (`build_patch_roughness.py`) | patch-scale `z0_kan`, `zd_kan`, `z0_kan_{θ}`, `flag_pai_over_envelope` |
| `features_roughness.parquet` ×8 | per-cell z0/zd, 4-method spread, validity flags |
| `rectangular_domain_v1.json` + `per_patch_indicators.csv` | domain extents, blockage ratio (~0.02 uniform), Blocken margins |
| `wind_rose.json` ×site | directional frequency + mean speed at z=10 m (annualisation weights) |
| `src/cfd_integration/README.md` | the inbound results contract + the 7 outbound coupling refinements |

## The tracks

`local` = computable now from morphometry + `buildings_extended_700m.gpkg`.
`gated` = needs a real OpenFOAM return.

| id | track | status | closes | effort |
|----|-------|--------|--------|--------|
| **W0** | **Floored z0 + flag + provenance** (the genuinely-morphometric core) | **DONE** `9a509d1` | skimming z0→0 / NaN log denominator; silent-staleness | S |
| **R-C** | MAR-P07×N pilot **requested** (drag-centroid + log-fit anchor) | **REQUESTED** | the validation gate — needs the one OpenFOAM return | — |
| **F** | Upstream-fetch inlet roughness `z0_inlet(θ)`, `zd_inlet(θ)` | deferred → pilot | council refinement #1 — "biggest coupling risk" | M |
| **I** | Inlet ABL fields: `u*`, `k`, `ε`/`ω`, derived turbulence intensity `I(z)` | deferred → pilot | replaces the assumed suburban `I≈0.15` with a per-patch value | S |
| **W-ks** | `ks = 9.793·z0/Cs` + two-zone ground (the rest of wall treatment) | deferred (CFD-owned) | council #2/#3 — but `Cs`/`yP` belong to the solver (review F6/W2) | S |
| **P** | Distributed canopy drag `Cd·a(θ)` for a porous / unresolved-fetch representation | deferred (ask CFD owner) | an *alternative* modelling path — unanswered interface question | M–L |
| **D** | Per-patch first-cell height (`yP > ks`) + refinement-zone extents | deferred (CFD-owned) | tidies meshing guidance; `yP` is a mesh quantity | S |

### Track F — upstream-fetch inlet roughness `z0_inlet(θ)`

**Compute the approach-flow roughness from the fabric the wind actually crosses before it
reaches the patch**, not the patch itself. For each patch and each of the 8 sectors θ:

- Build an **upwind fetch wedge** from `data/{site}/buildings_extended_700m.gpkg`: a sector
  (±22.5° about θ, or a rectangular fetch strip) extending upwind from the patch edge out to
  the 700 m context radius.
- On that fetch geometry compute the standard Kanda inputs — `λp`, `λf(θ)`, `H_mean`, `σH`,
  `H_max` — and run the same vendored UMEP/Kanda estimator as the patch path.
- Emit a **second column family** `z0_inlet_{N..NW}`, `zd_inlet_{N..NW}` distinct from the
  patch `z0_kan_{θ}`, plus the fetch provenance (`n_buildings_fetch`, `fetch_lambda_p_{θ}`).

The CFD side sets the Richards & Hoxey (1993) ABL inlet for a given run's direction from
`z0_inlet_{dir}` — the turbulence the patch *should see arriving* — and keeps the patch's own
buildings as resolved geometry. This is the fix for the double-count risk.

**Open sub-decision:** wedge vs. rectangular fetch strip; fetch length (patch-edge→700 m vs.
a fixed 500 m). Default: ±22.5° wedge, full available fetch to 700 m, area-weighted.

### Track W — wall treatment (floored z0, `ks`, two-zone ground)

The contract already specifies the physics (refinements #2, #3); this track *emits* it so the
CFD agent consumes numbers, not prose:

- **Floored z0** `z0_inlet_floored_{θ}`: `max(z0_inlet_θ, z0_floor)` with a documented
  `z0_floor` (candidate: 0.03 m, the suburban ground class already used for the empty-domain
  floor). Prevents the skimming-limit `z0→0` from blowing up the log profile. Flag which
  patches were floored.
- **Equivalent sand-grain roughness** `ks_inlet_{θ} = 9.793·z0_inlet_floored_θ / Cs` with the
  OpenFOAM `nutkRoughWallFunction` default `Cs = 0.5`, emitted for the **approach floor only**.
  Carry the near-wall constraint `ks < yP` as an explicit note + a computed max-`ks` given the
  contract's target `y+ ≈ 30–300`.
- **Two-zone ground table**: per patch, (a) rough **approach-floor** z0 = `z0_inlet_floored`
  in the clearing, (b) small **in-patch ground** z0 ≈ 0.01–0.03 m under resolved buildings
  (mesh-capped so `ks < yP`). One row, both zones, so "z0≈0 everywhere" can't happen by default.

### Track I — inlet ABL turbulence fields

One-line derivations from `z0_inlet` + the wind rose reference velocity, precomputed so the
CFD BCs are drop-in (Richards & Hoxey 1993 / OpenFOAM `atmBoundaryLayerInletVelocity`):

- `u*(θ) = κ · U_ref / ln((z_ref + z0_inlet_θ) / z0_inlet_θ)`, κ=0.40, z_ref=10 m,
  `U_ref` = the sector mean speed from `wind_rose.json`.
- `k = u*² / √Cμ` (Cμ=0.09); `ε(z) = u*³ / (κ(z + z0))`; `ω = ε / (Cμ k)`.
- **Derived turbulence intensity** `I_10(θ) = √(2k/3) / U_ref` — a per-patch, per-direction
  number that *replaces the contract's assumed suburban `I≈0.15`* with one consistent with the
  patch's own approach roughness. This is the cheapest credibility win in the set.

Emit as `patch_inlet_bc.csv` (per patch × direction) — the CFD-inlet BC sheet.

### Track P — distributed canopy drag (fork, exploratory)

For a **porous-media / distributed-drag** representation of the unresolved upstream fetch (an
alternative to resolving every upwind building), morphometry gives the sectional drag: a
canopy drag coefficient × frontal-area-density `Cd·a(θ)` derived from `λf(θ)` and `H_mean`,
usable as an `fvOptions` momentum sink or a `porosityProperties` Darcy–Forchheimer zone. Novel
but only worth building if the CFD repo wants a porous fetch rather than a rough-wall fetch.
**Deferred pending the Open-decision fork.**

### Track D — meshing guidance tidy

Emit per-patch first-cell height target from `ks` (`yP_min = ks / <fraction>`), refinement-zone
extent (= analysis disk + a buffer), and the expansion ratio into the hand-off manifest, so the
`snappyHexMesh` guidance is per-patch not one global paragraph. Low priority; the domain manifest
already carries extents and blockage.

## Sequencing (validate-first — as decided)

- **Now (DONE):** W0 floored z0 + flag + provenance on `patch_roughness.csv` (additive; the
  existing `z0_kan*` columns untouched) + the MAR-P07×N pilot request in the contract +
  the dashboard export card surfacing the hand-off. No `patch_inlet_bc.csv` yet.
- **Blocked on the pilot return (external, R-C):** when MAR-P07 comes back and R-C reports
  whether `kanda_precfd_v1` recalibrates, THEN build **F → I** per the hardened specs
  (F: new `src/morphometry/fetch.py` per the review's module split; I on the *floored* z0,
  coefficients not scalars) with the review's test list. `patch_inlet_bc.csv` lands here.
- **Ask-first / CFD-owned:** W-ks + two-zone (needs the solver's `Cs`), Track P (needs the
  rough-vs-porous requirement), Track D (needs `yP`). Resolve by a message to the CFD repo,
  not a guess.

## Open decisions — resolved

1. **Track P in or out?** → **Deferred; resolve by asking the CFD-repo owner** whether the fetch
   is represented rough-wall (Track F) or porous. It is an interface question, not a scope call.
2. **Fetch geometry for F (when built):** ±22.5° wedge, apex at patch center, annular sector
   `r=50 m → 700 m`, **bearing = the meteorological FROM direction** (the upwind-not-downwind
   test is the one guard against the symmetry-hidden flip — review Eng F2). Consider
   distance-decay weighting over uniform area-weighting. Pin the λf `dissolve` convention to
   the grid's.
3. **`z0_floor` value:** **fixed 0.03 m** (suburban ground class), NaN-safe, flagged per patch.
   Shipped in `roughness.floor_z0`.

## Guardrails

1. **Additive only** — new suffixed columns / new CSV; never mutate `z0_kan*` or any canonical
   morphometric column. (Repo contract.)
2. **No CFD execution here** — this repo produces the inlet+wall *specification*; OpenFOAM setup
   lives in the CFD repo.
3. **Decouple the two z0 roles** — inlet/approach (upwind fetch, rough) vs. in-patch ground
   (small, mesh-valid). Never double-count the patch's own buildings.
4. **Everything gated on `buildings_extended_700m.gpkg` existence** — degrade per-site if the
   fetch layer is absent (present for all 5 campaign sites today).
5. **Stage the test with the behaviour**; run the roughness/CFD test files, not the full suite.

---

## /autoplan review (2026-07-03) — 3 independent lenses, codex unavailable → `[subagent-only]`

Ran CEO (strategy), Eng (architecture/correctness), DX (consumer) lenses over this draft.
Consensus is single-model per phase (codex not installed); each finding is one lens unless noted.

### Consensus table

| Dimension | Verdict |
|-----------|---------|
| Right problem / sequencing sound? | **NO** — build validation (R-C pilot) before productionizing F/W/I (CEO, echoed by Eng E1) |
| Additive/decoupled framing correct? | YES (all three) |
| Track F spec complete? | **NO** — under-specified as a patch-path reuse; it is a new geometry pipeline (Eng F1) |
| Correctness failure modes covered? | **NO** — 3 plausible-but-wrong BC bugs uncaught: z0→0/NaN u\*, upwind/downwind flip, phantom towers (Eng I1/F2/F3, DX F3) |
| Hand-off self-service for the consumer? | **NO** — 6/9 journey steps need guesswork; ship a worked reference case (DX) |
| Track W scope correct? | **NO** — ks/Cs/yP is CFD-side; only floor+flag is genuinely ours (CEO F6, Eng W2, DX F5) |

### Cross-phase themes (≥2 lenses independently)

1. `z0=0/NaN → u*` division blows up for skimming/empty sectors (Eng I1 + DX F3) — **bug in this draft**.
2. `ks/Cs/yP` is CFD-owned and the chosen floor+Cs self-violates `ks<yP` (`ks≈0.59 m`) (CEO F6 + Eng W2 + DX F5/F6).
3. Fetch z0 inherits the same out-of-envelope Kanda invalidity, on a worse support (CEO F1/F2 + Eng E1).
4. No provenance/version stamp → silently stale after any re-baseline (CEO F3 + DX F8).
5. New asymmetric `z0_inlet(θ)` contradicts the symmetric-z0 caveat unless scoped to the patch column (Eng F2 + DX F7).

### Decision audit trail (auto-decided by the 6 principles; the sequencing call is NOT auto-decided → gate)

| # | Decision | Class | Principle | Rationale |
|---|----------|-------|-----------|-----------|
| 1 | Track I `u*` consumes `z0_inlet_floored`, not raw z0; assert `z0>0` | Mechanical | P5 explicit | Raw z0=0 → NaN u\*; this is a correctness fix, not a preference |
| 2 | Empty/NaN-fetch sectors use a NaN-safe floor branch, not `np.max()` | Mechanical | P1 completeness | `np.max(nan,0.03)=nan`; must branch on non-finite |
| 3 | Apply `phantom_mask` to fetch buildings before aggregation | Mechanical | P4 DRY | Reuse existing guard; known recurring `topo==0` corruption in this data path |
| 4 | Track F lives in a new `src/morphometry/fetch.py`; `roughness()` reused unchanged; BC algebra in `src/cfd_integration/inlet_bc.py` | Mechanical | P5 explicit | F is 90% geometry; keep the estimator untouched (additive guardrail) |
| 5 | Emit `ks` formula as **documentation**, not a committed synced column; ship floored z0 + flag as the owned deliverable | Taste→auto | P3 pragmatic | Cs belongs to the CFD solver; owning it invites drift (CEO F6, Eng W2) |
| 6 | Stamp `roughness_calibration="kanda_precfd_v1"` + git SHA + source gpkg in every hand-off file | Mechanical | P1 completeness | Trivial now, unreconstructable later |
| 7 | Ship one worked MAR-P07×N reference case (CSV row → `0/U`,`0/k`,`0/omega`,`0/nut` dicts) + constants/units header | Mechanical | P1 completeness | Collapses ~6 DX findings into one artifact |
| 8 | Carry fetch flags: `flag_fetch_empty`, `fetch_coverage_{θ}`, `flag_kanda_X_saturated`, fetch `flag_pai_over_envelope` | Mechanical | P1 completeness | Honesty; the fetch estimate is screening-grade |
| 9 | Scope the symmetric-z0 caveat to `z0_kan_{θ}`; document `z0_inlet_{θ}` asymmetry as legitimate approach-flow signal | Mechanical | P5 explicit | Otherwise the caveat reads as contradicting the deliverable (Eng F2, DX F7) |
| 10 | Defer Track P; resolve rough-vs-porous by **asking the CFD-repo owner**, not guessing | Mechanical | P6 action | It's an unanswered interface question, not a scope call (CEO F5) |
| — | **Sequencing: validate-first (pilot) vs build F/W/I now** | **User Challenge** | — | All 3 lenses recommend changing the stated "extend now" direction → gate |

### Revised NOT-in-scope (this cycle, pending the gate)

Track P (porous fetch — needs CFD-owner requirement); Track D mesh-sizing (CFD-owned `yP`); any *absolute* z0 confidence claim (CFD-gated); R-C (external OpenFOAM).

### Test plan (lands with any build path) — from the Eng lens

Priority tests: upwind-not-downwind on asymmetric fabric (the only guard against the symmetry-hidden flip); phantom-tower exclusion; empty-fetch NaN-safe floor; `ks` formula + feasibility flag; `u*` on floored z0 / rejects z0=0; `z0_kan` non-regression pin. Full 19-test list in the review transcript; land tests 1,4,5,9,13,17 at minimum before ship.
