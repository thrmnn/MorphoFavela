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
  — type predicts WHO-2h sun failure 14%→73%, transfers LOSO.
- [Variance: type vs site vs interaction](/outputs/cross_site/signature/figures_v2/index.html#fig-typology_variance)
  — morphotype 17% vs site 2% vs 0.7% interaction → transferable.
- [Block-scale morphotope tissue maps](/outputs/cross_site/signature/figures_v2/index.html#fig-morphotope_maps)
  — 5 tissues, distinct from the cell types; 4/5 recur.
- [TR coherence audit (punch list)](/outputs/_hub/docs/tr_audit.html)
  — the weaknesses being bulletproofed.

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
- **Remaining:** ⛔ ray-caster vs Radiance/SOLWEIG x-val — BLOCKED (neither installed
  locally; needs user). Optional: morphotype×site fingerprint heatmap (largely covered by
  `recurrence_evidence.png` / `composition_by_site.png` / `fingerprint_heatmap.png`).

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

1. **Typology → environmental-failure predictor** (implementing the plan):
   - ✅ Step 1 — per-type WHO-2 h failure lookup (`typology_failure_lookup.png`):
     monotone 14%→73%, T3→T4 regime jump, T4/T5 saturating.
   - ✅ Steps 2–4 — parsimony + LOSO transfer + calibration: type-only transfers
     out-of-site at **AUC-PR 0.77** (vs 0.85 full vector, 0.64 baseline; Δ0.086) —
     the discrete code keeps most of the signal at far lower dimension. Calibrated
     (slightly under-confident), PR AP 0.88 (`typology_parsimony.png`,
     `typology_calibration.png`).
   - ⏳ Next: isotonic recalibration; 3-level variance decomposition (between-type vs
     site vs site×type); blind risk map on the 3 calibration favelas (the payoff).
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
