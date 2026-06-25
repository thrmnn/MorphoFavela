# MorphoFavela → brisaverse — ROUND 4/5 results (2026-06-25)

Deliverables for the round-4 handoff (`morphofavela_handoff_2026-06-25_round4.md`)
and the round-5 extension embedded in `morphofavela_sync_2026-06-25.md`
("Commissioned from MF — round-5"). All produced this session on the
dissolved-λf canonical baseline. Source of every number below:
`scripts/brisa_ventilation/09_regime_taxonomy.py` →
`outputs/brisa_ventilation_fix/taxonomy_regime.json` +
`methods_morphometric_row.csv` (both regenerable; outputs/ is gitignored).

Commits: `1e5b67c` (keystone), `f999b0b` (fig04), `4ec9898` (lambda_f_regime),
`504c062` (morphotope composite), `1cdb3ac` (TR figure + PDF).

---

## R5 · Regime-stratified four-state taxonomy — CANONICAL, supersedes λf>2.75

Ventilation axis ≡ skimming flow regime, **dissolved λf > 0.65** (Grimmond &
Oke 1999). Sunlight axis unchanged (winter direct sun < 2 h). **Denominator =
n_classified = built ∩ sun-known ∩ λf-known = 56,631** (the canonical
street-solar denominator via nearest-3-within-25 m fallback — NOT the 64,389
raw built mask, NOT the 24,862 direct-support subset).

### Pooled + terrain (primary, λf > 0.65)

| group | n | adequate | sunlight | ventilation | compound |
|---|---|---|---|---|---|
| **Pooled** | 56,631 | 26.9 | 6.5 | 27.0 | **39.6** |
| Hillside | 26,573 | 26.8 | 10.3 | 20.8 | **42.2** |
| Flatland | 30,058 | 27.0 | 3.2 | 32.5 | **37.3** |

### Per-site (primary, λf > 0.65)

| site | n | adeq | sun | vent | comp |
|---|---|---|---|---|---|
| Vidigal | 2,548 | 20.7 | 13.0 | 22.8 | 43.4 |
| Rocinha | 7,637 | 11.9 | 7.6 | 13.7 | 66.7 |
| Complexo do Alemão | 16,388 | 34.6 | 11.1 | 23.7 | 30.6 |
| Rio das Pedras | 5,710 | 9.9 | 3.6 | 24.0 | 62.4 |
| Maré | 24,348 | 31.0 | 3.1 | 34.5 | 31.4 |

### ⚠ The "typology inversion" did NOT survive — it REVERSED

The deprecated λf>2.75 pre-screen gave **flatland 21.6 % > hillside 14.8 %**
compound. Regime-stratified gives the opposite: **hillside 42.2 % > flatland
37.3 %** (gap +4.9 pp). The λf-independent **H/W > 0.65 (Oke 1988) cross-check
agrees**: hillside 48.1 % > flatland 39.5 %. The old inversion finding must be
retired from decks + manuscripts; the corrected ordering is hillside-heavier
compound, robust across both the λf and H/W criteria.

### H/W cross-check (own sparser denominator n_classified_hw ≈ 55,768)

| group | n | adeq | sun | vent | comp |
|---|---|---|---|---|---|
| Pooled | 55,768 | 27.8 | 3.0 | 25.6 | 43.5 |
| Hillside | 26,438 | 21.1 | 4.4 | 26.5 | 48.1 |
| Flatland | 29,330 | 34.0 | 1.8 | 24.8 | 39.5 |

Compound clustering re-run on the regime mask (`compound_spatial_clustering.json`):
Moran's I **0.47–0.70**, all p=0.001; the contiguity finding holds *stronger*
than under the old mask (larger compound fraction → larger contiguous patches).

## R5 · Dissolved per-site Methods-table morphometric row

Medians + means over the classified denominator (`methods_morphometric_row.csv`):

| site | SVF med | λp med | **dissolved λf med** | H_mean med | σH med | slope med |
|---|---|---|---|---|---|---|
| Vidigal | 0.44 | 0.68 | 0.84 | — | — | — |
| Rocinha | 0.21 | 0.82 | 1.14 | — | — | — |
| Complexo do Alemão | 0.35 | 0.70 | 0.69 | — | — | — |
| Rio das Pedras | 0.15 | 0.93 | 1.17 | — | — | — |
| Maré | 0.35 | 0.82 | 0.83 | — | — | — |

(Full medians + means for SVF, λp, λf, H_mean, σH, slope are all in the CSV;
λf medians here are over the classified subset, ~equal to the full-mask values
in `lambda_f_regime.json`: Vid 0.83 · Roc 1.12 · Ale 0.68 · RdP 1.15 · Maré 0.81.)

## Figures re-exported (brisaverse deck assets)

- `lambda_f_regime.png` — R4-1 two-panel: Panel A per-site regime maps
  (isolated/wake/skimming, dissolved λf, CVD-safe luminance-ordered),
  Panel B regime-share strip (per-site + pooled, canonical shares). `4ec9898`.
- `morphotope_maps_repartition.png` — R4-2 combined k=3 tissue maps + per-site
  repartition bars (Compact Hillside / Mixed Dense / Saturated Flatland). `504c062`.
- `fig04_diagnostic_taxonomy.png` — R4-3 re-rendered on the regime axis (λf>0.65),
  nominal four-state, CFD-pending caption, H/W cross-check note. `f999b0b`.

All honour the round-3 §0 text-fit hard gate.

## SVF "6×" claim — VERIFIED, qualified (sync §4 action)

On the **pooled RF** (`rf_predictor_stats.json`, `pooled_rf.permutation_importance`):
SVF = 0.283, southness = 0.059, slope = 0.042. So SVF carries **~4.8× the next
feature (southness)** and ~6.7× slope. The "~6×" headline is defensible **only**
if it referenced SVF-vs-slope (or a mean-of-rest baseline), not the literal next
feature. The sync's "≈1.8×" used the *per-fold LOSO* permutation importances
(SVF ≈0.137–0.16), a different quantity from the pooled aggregate. Recommended
deck/manuscript wording: **"SVF's permutation importance is ~5× the next-ranked
geometric feature in the pooled model."** Qualitative "SVF dominates" holds.

## λf INTEGRATION READINESS — what is locked, what to watch

**λf is locked. Integrate from `lambda_f_canonical.json` (the single source).**
Generated by `scripts/brisa_ventilation/10_lambda_f_lockfile.py`, test-pinned
(`tests/test_lambda_f_lockfile.py`: §5.2 reproduction, 64,389 denominator,
dissolved<summed, skimming=0.65). Commit `51bbe33`.

### LOCKED ✅
- **Definition**: dissolved (party-wall corrected) — touching footprints unioned
  into blocks before projecting; `lambda_f_mean` = mean of the 4 distinct
  cross-wind axes. Pre-fix `lambda_f_mean_summed` preserved on every grid.
  Geometrically correct and unit-tested (`test_lambda_f_dissolve`: split block
  2.0→1.0). No external favela-λf benchmark exists — that absence *is* the
  paper's contribution, not a validation gap.
- **Canonical denominator** = full built mask `building_count > 0`, **pooled
  n = 64,389**. Use this for the λf descriptor / Methods table. Do NOT use the
  taxonomy's `n_classified` (~56,631) for λf — that denominator additionally
  requires a winter-sun observation and belongs only inside the four-state
  taxonomy.
- **Per-site dissolved λf (median · mean · over-count)**: Vid 0.826·0.880·1.59× ·
  Roc 1.117·1.168·1.96× · CDA 0.681·0.700·1.69× · RdP 1.152·1.172·2.34× ·
  Maré 0.815·0.848·1.97×.
- **Groupings (exact, both in the lock file — do not conflate)**:
  *signature_family* (TR §5.2 ternary): Hillside{Vid,Roc}=1.04 · Mixed{CDA}=0.68
  · Flatland{RdP,Maré}=0.87. *terrain_binary* (taxonomy): hillside{Vid,Roc,CDA}
  · flatland{RdP,Maré} (CDA folded into its hillside morro).
- **Flow regime** (full built mask): 65.2% skimming / 30.0% wake / 4.8% isolated,
  pooled median 0.83.
- **All 8 sites dissolved-consistent** (2026-06-25): the 5 campaign sites + the
  3 calibration sites (borel 1.28× · jacarezinho 2.07× · juramento 1.41×) were
  migrated. The calibration sites had been left on summed λf — now fixed, so the
  forthcoming blind-application predictor study uses matched features. Recovery
  recipe (gpkgs are gitignored): `python scripts/migrate_lambda_f_dissolve.py
  --site <name>` per site (deterministic).

### CONCERNS / ACTIONS for brisaverse to carry
1. **Retire the old "typology inversion" everywhere.** flatland>hillside compound
   is dead; the regime view reverses it (hillside 42.2% > flatland 37.3%, H/W
   cross-check agrees). Decks/manuscripts citing 21.6/14.8 must be corrected.
2. **Fix the "SVF 6×" headline** → "~5× the next-ranked geometric feature in the
   pooled model" (it is 4.8× vs southness, 6.7× vs slope; the 1.8× was per-fold).
3. **Denominator discipline**: any new λf statement must declare its denominator.
   The `methods_morphometric_row.csv` from script 09 uses the 56,631 taxonomy
   denominator (CDA median 0.692, not the canonical 0.681) — use the **lock file**,
   not that CSV, for published λf numbers.
4. **Compound-clustering JSON** (`compound_spatial_clustering.json`, Moran's I
   0.47–0.70) now rides on the regime compound mask — consistent with the
   manuscript's clustering claims, but re-cite from the regenerated file.

### Nothing else outstanding on λf for integration.
The number, denominator, groupings, regime, and over-count are all pinned and
reproducible. brisaverse can integrate now.

## Not done (out of scope this round, per handoff)

Solar figs (fig06–08), predictor fig (fig05) beyond the SVF text, roughness
envelope — untouched. fig04/lambda_f_regime are manuscript/deck assets, not TR
figures; the TR's own taxonomy is the CFD `U_mean<1.0 m/s` proposition
(`fig_0_4_diagnostic.png`) and its §4.5 regime paragraph is already on dissolved λf.

## ⚠ TYPOLOGY-AS-PREDICTOR — conclusion FLIPPED under the dissolved-λf re-baseline

Re-running `scripts/analyze_typology_predictor.py` + `typology_predictor_extra.py`
on the dissolved-λf features (the morphotypes were re-fit on dissolved λf) **reverses
two earlier headlines**. Do NOT cite the pre-re-baseline parsimony story.

- **Parsimony reversed.** Type-only leave-one-site-out **AUC-PR fell 0.77 → 0.61**
  (baseline prevalence 0.56), and the gap to the continuous fabric vector widened
  **+0.086 → +0.229** (vector 0.84, both 0.84). The discrete typology alone is now
  only *modestly* skillful; the **continuous fabric vector carries the transferable
  signal**. Retire "the discrete code keeps most of the signal."
- **Variance partition flattened.** Two-way partition of cell WHO-2 h failure:
  morphotype **6%** ≈ site **7%**, site×type **0.5%**, residual **87%**. Retire
  "morphotype dominates." The honest read: type and site contribute comparably and
  modestly; the negligible interaction means the (small) type signal still transfers,
  but most cell-level variance is within-type.
- **Calibration / isotonic.** The raw LOSO `both` model is *already* well-calibrated
  (ECE 0.018); isotonic recal adds nothing (0.018 → 0.023). Report the raw model.
- **Blind risk map stands** (`typology_blind_riskmap.png`): the model never saw
  borel / jacarezinho / morro_do_juramento; morphology-only mean p̂ = 52 / 63 / 55 %.
  Frame as **survey/simulation prioritisation**, not household-level risk — the
  modest type-only skill is exactly why.
- Figure titles for parsimony/variance/isotonic are now **data-driven** (computed
  from the numbers), so they can't re-assert a stale story on the next regen.
- Cause of the break that surfaced this: the re-baseline dropped `party_wall_ratio`
  from `features_grid` (it was never a grid column — separate per-building analysis;
  the dissolve now encodes party walls into λf). Both predictor scripts were repaired
  (5-feature continuous vector) + a schema-drift guard test added.

**Net for the paper:** the typology is a sound *descriptive/communication* device and
a usable coarse prioritiser, but the **continuous morphometrics are the predictor**.
This is more defensible than the over-claimed parsimony story it replaces.

## Remaining MorphoFavela-side tracks

- **Track B figure reworks — ✅ DONE** (this session): fig03 regime axis +
  DELIVERED/PROVISIONAL asymmetry; fig05 AUC honesty box + sun-adequacy-floor target;
  fig01 Panel C terrain-hillshade / building-height-ramp split. All gate-checked.
- **Track E — ✅ DONE**: lateral-connectivity scalar (`scripts/run_lateral_connectivity.py`,
  `lateral_connectivity.{json,png}`). Per built cell, the distance-transform depth into
  contiguous fabric = the pre-CFD LATERAL ventilation-tendency signal, companion to λf's
  VERTICAL regime. Pooled median 32 m; **flatland is doubly constrained** — RdP (med 42 m)
  and Maré (140 m+ deep cores) sit far from any opening *and* dominate the λf skimming
  regime, while the perforated hillside fabric (Vidigal 22 m) is laterally shallow. The
  double-constraint is real at the **cell** level, not just site-level: pooled Spearman
  **ρ(open_edge_dist, λf) = +0.49** (p≈0; per-site +0.32…+0.53), and **42 % of built cells
  are both deep (≥ median) and skimming (λf ≥ 0.65)**. Both signals are geometry-only
  tendencies; neither delivers adequacy (τ CFD-gated).
- **Track F** (ongoing data-quality sweeps): fixed the `print3d` extrude latent break
  (hardcoded missing `triangle` backend → engine-agnostic + force_2d) and the
  `party_wall_ratio` schema drift.
- Blocked-isolated (skip, don't stall): ray-caster vs Radiance/SOLWEIG, CFD-τ,
  Mingze WeTransfer upload.
