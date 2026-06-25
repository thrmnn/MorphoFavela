# Work queue & latest

The living board for this project — what just shipped, what's in progress, what's
queued, and what's blocked. Surfaced at the top of the hub so new results are never
hard to find. Updated every working session.

## 🆕 Latest (each links to the exact figure/section)

- [TR §6.6 roughness — invalidity caveat](/outputs/_hub/docs/technical_report.html#66-aerodynamic-roughness-z0-zd)
  — per-cell z0/zd invalid 53–75%; the envelope is the result.
- [TR §5.5 Morphological Typology & Signature](/outputs/_hub/docs/technical_report.html#55-morphological-typology-signature)
  — morphotype (cell) vs morphotope (tissue), two distinct levels.
- [Typology → environmental failure (money figure)](/outputs/cross_site/signature/figures_v2/index.html#fig-typology_failure_lookup)
  — per-type WHO-2h sun-failure rate; continuous fabric vector is the predictor
  (type-only LOSO AUC-PR 0.61 vs 0.84), typology = descriptive/coarse-prioritiser.
- [Variance: type vs site vs interaction](/outputs/cross_site/signature/figures_v2/index.html#fig-typology_variance)
  — morphotype 6% ≈ site 7%, interaction 0.5%, residual 87% (re-baselined).
- [Block-scale morphotope tissue maps](/outputs/cross_site/signature/figures_v2/index.html#fig-morphotope_maps)
  — 5 tissues, distinct from the cell types; 4/5 recur.
- [TR coherence audit (punch list)](/outputs/_hub/docs/tr_audit.html)
  — the weaknesses being bulletproofed.

## 🟢 λf dissolve re-baseline — Track A ✅ COMPLETE (2026-06-25, full autonomy)

Operating charter: `docs/autonomous_execution_plan.md`. User chose **full switch to
dissolved λf + full morphotype re-baseline + data-driven k**. **ALL DONE + committed
(4a6ed1c→97b69ac):** core `dissolve=` option (tested); 5 grids migrated (over-count
1.45–2.10×, `lambda_f_mean_summed` preserved); regime classifier (65% skimming/30% wake);
**k=6 cell morphotypes re-fit** (LOSO ARI 0.763, CH peaks k=6) + **re-named** (T0 Open
Fringe, T1 Flatland Consolidated, T2 Hillside Fringe, T3 Shaded Consolidated, T4 Hillside
Core [universal 5/5], T5 Saturated Core [T1/T5 flatland-conditional]); **k=3 morphotopes**
(data-driven, was 5; bootstrap ARI 0.916) = Compact Hillside / Mixed Dense / Saturated
Flatland Tissue; roughness z0/zd regen on dissolved (RC-1 invalidity 53–75% HOLDS);
predictor regen (svf-driven, robust, β−2.03); **statsmodels 0.14.6 installed** (pinned,
ABI intact); **TR §4.2/§4.5/§5.5/§6.6 rewritten + PDF rebuilt (30.5 MB) + 5 figures
refreshed** (typology table 1.04/0.68/0.87; λf>2.75 dropped → Oke/GO regime). Whole stack
consistent + green (648 passed).

## 🟢 Round 4/5 — regime-stratified re-export ✅ COMPLETE (2026-06-25)

Handoff `morphofavela_handoff_2026-06-25_round4.md` + round-5 extension in the
sync. Results-back: `docs/brisa_round4_results_2026-06-25.md`. Both PI decisions
adopted (dissolved λf canonical; taxonomy ventilation axis = skimming regime).
- ✅ **① Regime taxonomy keystone** (`scripts/brisa_ventilation/09_regime_taxonomy.py`,
  `1e5b67c`): dissolved λf>0.65 primary + H/W>0.65 cross-check, **56,631** canonical
  denominator, methods morphometric row. **The typology inversion REVERSED** —
  hillside 42.2% > flatland 37.3% compound (was flatland 21.6% > hillside 14.8%);
  H/W cross-check agrees (48.1% > 39.5%). Pooled compound **39.6%**.
- ✅ **③ fig04** re-rendered on regime axis (`f999b0b`); clustering re-run on regime
  mask (Moran's I 0.47–0.70, holds stronger).
- ✅ **④ lambda_f_regime.png** → two-panel maps + share strip (`4ec9898`).
- ✅ **⑤ morphotope_maps_repartition.png** combined k=3 asset (`504c062`).
- ✅ **⑥** SVF-"6×" verified: pooled RF SVF=0.283 = ~4.8× next feature (southness),
  ~6.7× slope; "6×" only vs slope; sync's 1.8× was per-fold. Brisaverse-side fix.
- ✅ **⑦** TR morphotope figure title M0–M4→M0–M2 + PDF rebuilt (`1cdb3ac`);
  TR already dissolved-consistent (§4.5 regime, §5.5 k=3, λf table). Numerical audit PASS (13/13).
- ✅ **λf LOCKED for brisaverse integration** (`51bbe33`): `lambda_f_canonical.json`
  (script 10) = single source — canonical denominator (full built mask, n=64,389),
  per-site median/mean/over-count, both groupings exact (signature_family §5.2
  1.04/0.68/0.87 reproduced; terrain_binary taxonomy), regime shares; test-pinned.
  **All 8 sites now dissolved-consistent** — the 3 calibration sites (borel/
  jacarezinho/juramento) were still on summed λf and were migrated. Integration
  readiness + concerns documented in `docs/brisa_round4_results_2026-06-25.md`.

## 🟡 Track B — figure-regen rounds 2/3 (STARTED 2026-06-25)

Handoffs: `docs/brisa_figures_handoff_round{2,3}_2026-06-24.md`. ✅ Round-3 §0
**text-overflow HARD GATE** landed (`fig_style.check_text_overflow` + `save_fig(gate=True)`,
tested, commit 0158731) — every rework calls it before export. fig04 + lambda_f_regime
already re-rendered under round 4/5 above (regime-nominal + gate). **Remaining (my-call, run
via subagents, review each PNG):** fig04 (round-3 regime-based **nominal** taxonomy, NOT
the 4-state heat ramp; hatch-provisional; anti-misuse banner; F/G/H 3-col reflow; Okabe-Ito
luminance-ordered), fig03 (regime panel C from `lambda_f_regime`; DELIVERED/PROVISIONAL
asymmetry; λf maps→regime maps), fig01 (2×2 nominal taxonomy matrix; 3-D terrain-hillshade
vs building-height-ramp separation), fig05 (AUC honesty box full-sample 0.76–0.84 + CC
0.87–0.93; separable PD; "P(below adequacy floor)"). Then Tracks **D** regime→fig03, **E**
lateral-connectivity scalar, **F** data-quality sweeps. Blocked-isolated: ray-caster
(Radiance/SOLWEIG), CFD-τ, Mingze upload.

## 🔄 Autonomous loop (Workflow-driven, 2026-06-19)

Persistent mechanism = the **Workflow** tool (parallel agents, background, notifies on
completion). Pre-authorized: auto-merge green branches, my-call on names/palette/figures,
local compute, **no CFD**, no HPC. **Batch 1 DONE + integrated** (4 parallel agents, 583 tests green): finish the
typology predictor (isotonic recal + blind risk map on the 3 calibration favelas), cell-k
rigor (BIC/silhouette/CH/DB + LOSO ARI), morphotope-k bootstrap stability, street-network
/ beco metrics. Each returns a self-contained deliverable; the main session integrates →
gallery → hub → commit. Blockers removed: **consolidated env** (environment.yml, blocker
#1); **names + Okabe-Ito palette FINAL** (D19).

## 🖨️ 3D-print track (`src/print3d/`, on main)

Patch → watertight scaled STL (terrain plinth + embedded buildings + manifold3d union;
1:1000 → 10 cm). Built: ROC-P18 (steep hillside), MAR-P20 (flat fabric), **RDP-P20 +
VDG-P07 (user-requested)** — in gitignored `outputs/{site}/print/`. Next: 1:500 re-emits
on request; one print per morphotype (physical typology set); hub print gallery.

## 🎨 Brisaverse figure tasks (`docs/brisa_paper_figures_handoff_2026-06-24.md`)

Canonical pipeline `outputs/paper_figures/*.py`; run on **IVF**. Provenance rule: ALS
heights=MIT/SondoTecnica, cadaster=IPP — pipeline open, data NOT redistributable.
- ✅ **PI reworks DONE (2026-06-24):** fig04 severity 4-colour palette (blue→yellow→
  orange→red) + per-site share **side table** + **(G) Moran's I panel** and the new
  **compound-state spatial-clustering** statistic — Moran's I **0.33–0.51 (all p<0.05)**,
  BB join-count z **27–112**, **53–92%** of compound cells in clusters ≥500 m² → fills the
  manuscript `\TODO` ("compound states cluster" now evidence). Stats →
  `outputs/brisa_ventilation_fix/compound_spatial_clustering.json`. fig03 C/D re-rendered
  as filled per-site coloured densities (`SITE_COLORS`) with hatched failing mass
  (bimodality / mass-at-zero now visible). fig05 audited (PASS — top-4 + sign logic
  correct) and recoloured: family-grouped bars (A), family-hued PD (B), sign-diverging
  coefficient markers (C). Test `tests/test_compound_clustering.py` (3 cases). PNGs in
  `outputs/paper_figures/exports/` for brisaverse to promote.
- ✅ **fig01 composite DONE (2026-06-24, commit 24ea0e4):** `fig01_composite.py` — A
  pipeline schematic + B Rio site map/insets + C four 3-D STL massing excerpts +
  provenance footer. Kept SEPARATE from the TR's map-only `fig01_study_sites.py`
  (figure-tracks convention); brisaverse promotes `exports/fig01_composite.png`. Also
  reconciled the C. do Alemão **mixed-vs-hillside** label (fig04 now reads canonical
  `SITE_TYPES`).
- ✅ **Full-sample LOSO DONE (2026-06-24, commit 80cde80) — handoff premise CORRECTED.**
  Aspect imputation is a **no-op** (aspect ~complete, ≤8 NaN); the real 56,631→22,238
  drop is **SVF ~50% + street-entropy ~54%** (street-sampled). Imputing SVF (top
  predictor) would inflate, not bound, the AUC. Honest inverse check instead:
  reduced-feature (no SVF/street-entropy) LOSO on the **full 56,631 cells** →
  **AUC 0.75–0.84 (mean 0.78)** vs complete-case full-feature **0.87–0.93**; the ~0.12
  gap is the SVF contribution on street-adjacent cells. Full prevalence 46% vs CC 56%
  confirms the coverage bias. `scripts/run_fullsample_loso.py` + test →
  `outputs/paper_figures/fullsample_loso.json`.
- ✅ **λf neighbourhood supplement DONE (2026-06-24, commit 8cef644 — user chose
  non-destructive supplement).** `scripts/run_lambda_f_neighbourhood.py` aggregates cell
  λf to a 100 m window (Σfrontal/Σplan = proper Grimmond–Oke λf). Finding: neighbourhood
  λf still ~1–3 (pooled median **1.65**); **~96% of fabric past the 0.35 skimming onset**,
  densest Rio das Pedras 2.6 / Rocinha 2.2 → favela fabric uniformly skimming-flow,
  exceeds textbook 0–1.5 even at neighbourhood scale (stronger E2). cell-scale fig03/fig04
  UNCHANGED. → `lambda_f_neighbourhood.json` + `exports/lambda_f_neighbourhood.png` + test.
- ✅ **Task 8 morphotype×site fingerprint DONE (commit a228e76):**
  `figures_v2/type_site_fingerprint.png` — commonality-forward composition heatmap (full
  type names + per-type "in N/5 favelas" recurrence count). T0 & T5 universal, T1/T4 in
  4/5, T2/T3 terrain-conditional → shared recurrent type set for P4/E2.
- ⛔ **Task 7 ray-caster x-val — BLOCKED** (no Radiance/SOLWEIG locally; needs user).
- ⚠️ **λf AUDIT (commit a184b4d, user-requested):** directional averaging CORRECT (N≡S/E≡W
  exact); stored grid λf reproducible (99.8% match clipped recompute; ~12 stale phantom
  cells). BUT `compute_frontal_area_ratio` **sums cadastral footprints → counts party
  walls** → ~**2.5× over-count** in fused fabric (summed cell median ≈1.6 vs dissolved
  ≈0.65). Aerodynamic λf is ~2–2.5× lower but still >0.35 skimming (E2 holds, magnitude
  inflated). **USER DECISION:** pipeline-wide dissolve fix touches fig03/fig04/predictor/
  roughness — deferred. **Brisa handoff: tasks 1–6 + 8 DONE; only 7 blocked.**

## ⚙️ Env (2026-06-24): work on IVF; consolidated env deferred

Both consolidation paths failed (unsolvable pins; clone+extras broke scipy ABI). Broken
clone removed. **IVF is the working env** (has esda/sklearn/geopandas/matplotlib/seaborn/
trimesh/manifold3d). Gaps: spopt/umap (Batch 2), statsmodels (done). Retry recipe in
`docs/autonomous_loop_plan.md`.

## 🧪 TR audit (ongoing bulletproofing — `docs/tr_audit.md`)

- ✅ Critical fixes: §6.6 renumber (was 2nd §6.5), roughness-invalidity + 180°
  caveats, morphotype/morphotope disambiguation, recurrence reconciled, predictor
  marked forthcoming, validation + validity figures embedded.
- ✅ Exec Summary bullets (signature + roughness) + §10.7/§10.8 limitation
  subsections added; version bumped.
- ☐ Remaining (low): §12 reproducibility Stages 7–8; §5.5 heading hierarchy; the
  pre-existing missing `fig5_wind_panel.png` ref.

## ✅ Shipped this session

- **Technical report §5.5 Morphological Typology & Signature** — added with
  schematics / composition / morphotope figures; PDF rebuilt.
- **Hub "Latest & work queue" panel** — newest results with direct links + this board.
- **Typology-as-predictor plan** drafted (`docs/typology_predictor_plan.md`).
- **5 morphotopes named** (Stepped Hillside … Flat Dense Core).

## 📋 Queued (prioritized)

1. **Typology → environmental-failure predictor** — ✅ COMPLETE, with a
   **CONCLUSION FLIP** under the dissolved-λf re-baseline (full detail in
   `docs/brisa_round4_results_2026-06-25.md` ⚠ section; commit `2e8d4b5`):
   - Parsimony REVERSED: type-only LOSO **AUC-PR 0.77→0.61**, gap to the continuous
     vector **+0.086→+0.229** (vector 0.84). The continuous fabric vector carries
     the transferable signal; the discrete typology is a descriptive/coarse-
     prioritiser. "Discrete code keeps most of the signal" RETIRED.
   - Variance partition flattened: morphotype 6% ≈ site 7%, interaction 0.5%,
     residual 87%. "Morphotype dominates" RETIRED. Figure titles now data-driven.
   - Isotonic recal: raw LOSO already well-calibrated (ECE 0.018→0.023, no gain).
   - Blind risk map DONE (borel/jacarezinho/juramento mean p̂ 52/63/55%).
   - Surfaced via a `party_wall_ratio` schema-drift break; both scripts repaired
     (5-feature vector) + drift-guard test added.
2. **Street-network / beco metrics** — 2nd configuration feature (circulation reach,
   alley width).
3. **Terrain-following morphometry** — fix the roughness datum confound (council C).
4. **Bootstrap-stability on the morphotope k** + k-selection rigor for the cell
   morphotypes (BIC/ICL + leave-one-site-out ARI).
5. **Draft the signature + roughness sections into the brisaverse paper.**

## ⛔ Blocked / gated (needs the user or external)

- **R-C CFD roughness validation** — needs real OpenFOAM (synthetic placeholders only).
- **Merges to `main`** — each track branch merges on user OK.
- **Mingze HTML report refresh** (#41) + **20 m re-baseline** (#24) — backlog, user call.

## Workflow notes

- Every track runs on a `track/*` branch; build → test → ruff → commit → push →
  refresh the hub, autonomously; merge to main on user OK.
- **Keep the technical report in sync** — when a morphology/signature/roughness
  visualization changes, update `technical_report.md` + rebuild the PDF in the same
  commit, and copy the figure into `docs/technical_report/figures/`.
- This queue is the single source of truth for outstanding work; update it as items
  move.
