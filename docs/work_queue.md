# Work queue & latest

The living board for this project — what just shipped, what's in progress, what's
queued, and what's blocked. Surfaced at the top of the hub so new results are never
hard to find. Updated every working session.

## 🆕 Latest (review these)

- **Block-scale morphotopes** (2026-06-19) — the favela signature at ~50 m tissue
  scale; 4/5 tissues recur, resolves the T2/T3 critique. → gallery "Morphotopes"
  group + overview "Validation 3".
- **Configuration: party-wall adjacency** — favela fabric fused everywhere
  (0.6–1.0 vs ~0.1 detached); flat types fully party-walled, hillside more stepped.
- **Designer morphotype schematics + per-favela maps + composition %** — in the
  overview and gallery.
- **Roughness physical-validity reckoning** — per-cell z0/zd invalid in 53–75% of
  cells; lead with the method envelope (decision A adopted).

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
