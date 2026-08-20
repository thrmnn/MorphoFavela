# λf dissolve-fix + figure-regeneration plan (2026-06-24)

Forward plan after the λf audit + dissolve prototype, and the brisaverse
round-2/round-3 figure handoffs. Decisions that need **you** are tagged ⟶DECIDE.

## 1. What the λf audit + prototype established

- **Directional averaging is correct** (N≡S/E≡W exact; `lambda_f_mean` = mean of
  the 4 cross-wind axes). Stored grid λf reproduces a clipped recompute (99.8%).
- **`compute_frontal_area_ratio` over-counts ~1.7×** (median; p90 3.2×) because it
  sums *cadastral* footprints, counting internal party walls in fused fabric.
  Dissolving footprints per cell → λf median 2.19→**1.12** (Rocinha).
- **The dissolve does NOT un-saturate the ventilation axis**: ~90% of the fabric is
  still past the 0.35 skimming onset even with corrected λf. The round-3
  "knife-edge" saturation is a **real fabric property**, not an artifact.
- ⟹ The dissolve fix matters for **absolute-λf consumers (roughness z0/zd, the
  Grimmond–Oke regime statement)**; the **multi-variable ventilation index**
  (round-3 §1) is still needed for axis *discrimination*. Complementary.

Prototype: `scripts/prototype_lambda_f_dissolve.py` (rocinha review PNG +
`lambda_f_dissolve_compare.gpkg`); primitive tested in `tests/test_lambda_f_dissolve.py`.

## 2. λf dissolve-fix rollout (pending your review of the prototype)

1. **Validate on all 5 sites** — run the prototype per site; confirm the ~1.7×
   factor and the persistence of saturation. (~1 min/site.)
2. **Promote the primitive into the core** — add `dissolve=False` to
   `compute_frontal_area_ratio` (per-block silhouette × area-weighted height),
   with the prototype's tests. Add `lambda_f_*_dissolved` columns to the grid
   **alongside** the summed columns (non-destructive).
3. **Re-wire consumers by what each needs:**
   - **Roughness `build_roughness.py` (z0/zd)** → switch to dissolved λf
     (absolute frontal density is the physical input). Regenerate roughness
     outputs + TR §6.6. ⟶DECIDE: confirm dissolved is canonical for roughness.
   - **fig03/fig04 cell threshold (2.75 relative p75)** → the relative p75 is
     roughly invariant to the dissolve (monotone-ish), and the saturation
     persists, so the diagnostic maps barely change. ⟶DECIDE: keep summed for
     the cell diagnostic (least churn) or move both to dissolved for consistency.
   - **Predictor** → λf is a weak feature (importance 0.029); regenerate with
     dissolved for cleanliness, no headline impact.
4. **Re-baseline the neighbourhood-λf supplement** on dissolved λf (its current
   numbers carry the over-count caveat).

## 3. Ventilation-axis decision (round-3 §1, REVISED 2026-06-24) — DECIDED

The urban-physics council **REJECTED the multi-variable index** (H/W collinear
with SVF on the vertical-openness/solar axis → "openness measured three times",
unfittable without ground truth). The adopted position is exactly what the
dissolve prototype pointed to: **λf saturation is the FINDING, not a liability.**

- **Tier 1 — geometry classifies the flow REGIME at scale** (no fitting, no
  per-cell adequacy claim): λf vs the Oke/Grimmond–Oke thresholds — **isolated
  ≲0.15 · wake 0.15–0.65 · skimming ≳0.65**. The fabric is *uniformly skimming*
  (the most ventilation-suppressed canopy class) = the at-scale result. **Use the
  dissolved λf** so the regime statement rests on the honest magnitude (dissolved
  cell median ≈1.12, neighbourhood ≈1.0 — still > 0.65 skimming onset, so the
  conclusion is robust to the over-count correction).
- **Tier 2 — CFD age-of-air τ carries per-cell ventilation ADEQUACY** on a
  validated patch sample; only τ may claim "adequacy". Validation = Spearman/
  Kendall rank concordance (geometry rank vs τ), patch sample stratified on the
  JOINT geometric distribution. Gated on the CFD campaign (out of scope here).
- **Drop** the λf>2.75 percentile flag + the 6.6% "ventilation-only" state
  (report as one sentence). Axes stay orthogonal single-scalars: SVF→solar,
  λf→regime label. The only genuinely-independent pre-CFD ventilation signal is
  **lateral connectivity / exposure-to-open-edge** (distance-to-open-boundary) —
  the quantity that still varies once λf plateaus; qualitative, superseded by τ.

**Next analyses (council "do now"):** (a) the λf **flow-regime classification**
(cheap — shares per band, per site, on dissolved λf); (b) optionally the single
lateral-connectivity scalar; (c) patch-CFD rank validation (CFD-gated). The
dissolve fix (§2) is the honest λf input to (a).

## 4. Figure regeneration (round-2 P0/P1/P2 + round-3 §0–§5)

Cross-cutting first:
- **Text-fits-in-box HARD GATE** (round-3 §0): a reusable assertion pass
  (`renderer.get_window_extent` ⊂ container − inset) that fails the export on any
  overflow. Build once, apply to fig01/fig03/fig04.
- **Palette:** Okabe-Ito, **luminance-ordered, nominal (not a heat-ramp)**; off
  saturated red; the four-state palette shared verbatim by fig01 matrix + fig03.
- **Firewall in the pixels:** hatch provisional ventilation-bearing states;
  in-panel anti-misuse banner; OWED legend phrasing; site order hillside→flatland.

Per figure:
- **fig04 (Fig 3):** nominal palette + hatch provisional + anti-misuse banner +
  table (H) → "largest contiguous upgrading unit (m²)" + (G) "(provisional axis)"
  + 3-column F/G/H reflow (no gutter content).
- **fig03 (Fig 2):** DELIVERED(solar) vs PROVISIONAL(λf) asymmetric styling +
  regime strip on the λf colourbar + box/violin strip panels C/D (preserve the
  0 h bin) + rasterized maps.
- **fig01 (Fig 1):** 2×2 nominal taxonomy matrix + ALS-derived-DTM elbow +
  single global provenance footnote + 3-D terrain(hillshade)/building(height-ramp)
  separation + height colourbar.
- **fig05 (Suppl):** AUC honesty box (full-sample 0.76–0.84 **and** complete-case
  0.87–0.93, prevalence 46% vs 56%) + separable PD lines + "P(below adequacy
  floor)" + shrink the pending-CFD stub.

## 4b. Morphotype-cascade stability (measured 2026-06-24) — DECIDED 2026-08-20: FULL RE-BASELINE

**PI ruling (2026-08-20, interview):** Option B — full re-baseline. Regenerate
`features_grid.parquet` on dissolved λf, re-fit the k=6 morphotype GMM, and
cascade through type names, recurrence, the typology predictor, TR §5.5, and
the brisa P4/E2 figures. The ARI 0.226 churn is accepted: the published types
re-fit on the corrected physics rather than preserving continuity with the
summed-λf artifact. Execution is a tracked campaign (bootstrap-ARI 0.90
stability bar applies to the NEW fit; old typology archived, never deleted).
Original decision block preserved below for the record.

### (superseded decision block)

The "full switch" ripples into the morphotype GMM (`lambda_f_mean` + `lambda_f_aniso`
are 2 of the 6 signature features). Non-destructive check (re-fit k=6 on the
SAME 64,355 built cells, summed vs dissolved λf): **ARI = 0.226**, only 58% of
cells keep their majority type. The dissolve is NOT a benign monotone rescale at
the clustering level — it **re-defines the published types** (the track's own
stability bar is bootstrap ARI 0.90). Cascading would churn type names,
recurrence, the typology predictor, TR §5.5, and the brisa P4/E2 figures.

State is currently SAFE: `grid_metrics` carries dissolved λf (+ summed preserved),
but `features_grid.parquet` is NOT regenerated, so morphotypes / roughness /
predictor still run on summed λf — nothing downstream has shifted yet.

**Fork (⟶DECIDE):**
- **A — Scoped switch (recommended):** dissolved λf canonical for absolute-physics
  consumers (roughness z0/zd, regime classification — done); morphotype signature
  stays on summed λf (point the signature at `lambda_f_mean_summed`). Published
  typology/predictor/TR §5.5/figures unchanged. Rationale: morphotypes are a
  *relative* clustering; the over-count is the issue for *absolute* λf, which is
  exactly where the fix is applied.
- **B — Full re-baseline:** regenerate `features_grid` on dissolved λf, re-fit the
  morphotypes (new typology, ARI 0.23 vs current), re-name, re-validate, cascade
  through recurrence + predictor + TR §5.5 + brisa figures. The "most-correct
  feature" position, but a large invalidation of published work.

## 5. Still blocked / PI-level
- ⛔ Ray-caster vs Radiance/SOLWEIG — not installed locally.
- ⟶DECIDE (round-2): full lexical rename of taxonomy states to "Below … floor
  (owed)" across the manuscript body; pull SVF–sun decoupling into Fig 2 vs ED.

## Suggested order
Review prototype (you) → §2.1 validate 5 sites → §2.2 core integration → §3 index
(if adopted) → §4 figure regen against the new index. §4 cross-cutting gate can
proceed in parallel since it's mechanical.

## 6. DECIDED 2026-08-20 (via /ops card, option A) — re-pin stability bar to measured 0.843

**PI ruling (2026-08-20T20:43Z, /ops decision morphofavela.ari-stability-bar
= A):** re-pin the stability bar to the measured bootstrap ARI 0.843
(sd 0.185, min 0.526, n=20, 5-site CAMPAIGN_SITES population, k=6, seed 0)
and proceed with the cascade (phases 2-3). TR §5.5's stability claim is
corrected to the honest measured number with its config stated; the
unreproducible 0.90 is retired. Original decision block preserved below.

### (superseded decision block — resolved by the ruling above)

Phase-1 of the full re-baseline (§4b, Option B) re-ran
`build_feature_table.py` → `build_signature.py --k 6` →
`refine_signature_spatial.py` → `pytest`. The k=6 fit itself reproduces the
2026-06-25 dissolved-λf fit to float noise, but the bootstrap-stability gate
(`outputs/cross_site/signature/stability_meta.json`) measures **0.842**
(sd 0.174, min 0.542, n=20), against the track's own **0.90** bar (the number
this plan's §4b PI ruling quoted as the bar the new fit must clear). Full
numbers and root-cause analysis in `docs/morpho_signature_decisions.md` (D24).
UPDATE (same day): the suspected site-scope bug in
`refine_signature_spatial.py` (pooling all 8 sites incl. 3 out-of-scope
calibration sites) was FIXED — bootstrap population now filtered to the
5-site `CAMPAIGN_SITES`, matching `build_signature.py` and decision D17 —
and the gate re-measured: **0.843** (sd 0.185, min 0.526). The pooling bug is
RULED OUT as the cause; the k=6 dissolved-λf fit's bootstrap stability on
the correct population is genuinely ~0.84, below the 0.90 the TR §5.5 quotes.
Where 0.90 was measured from is unestablished (possibly a different
feature-table state or bootstrap config on 2026-06-25).

**Options:**
- **(A) Re-pin the bar to the measured 0.843** and proceed with the cascade
  (phases 2-3), correcting TR §5.5's stability claim to the honest number
  with its config stated. Recommended: the fit is bit-reproducible; 0.84
  mean with min 0.53 is a describable stability profile, and the 0.90 claim
  cannot currently be reproduced.
- **(B) Investigate before cascading**: hold phases 2-3; probe why bootstrap
  draws vary (k choice, feature scaling, n_boot=20 too small) and whether a
  config change recovers ≥0.90 defensibly.
- **(C) Abandon the re-baseline**: keep the 2026-06-25 published fit as-is
  (it is what this run reproduced anyway) and only fix the TR's stability
  number.

Phase 2+ (morphotope, typology predictor, TR §5.5, brisa figures) is
**halted pending this decision** — nothing downstream has been re-run.

**Fork:**
- **Accept the lower stability bar.** 0.84 is still a strong bootstrap ARI in
  absolute terms (well above chance, comparable to the original LOSO-ARI 0.76
  the k=6 choice was defended on in the first place). Document the revised bar,
  proceed to phase 2 (morphotope/predictor/TR/figures) on the current fit.
- **Investigate before proceeding.** Two sub-questions worth separating: (a)
  is 0.90 even the right number to hold the *dissolved*-λf fit to — it was
  measured on the *summed*-λf fit, a different feature distribution; (b) fix
  the `refine_signature_spatial.py` scope bug (filter to `CAMPAIGN_SITES`) and
  re-measure, since the current 0.842 is itself contaminated by out-of-scope
  calibration-site data of unknown influence direction/magnitude.

Owner: PI. Blocks: morpho_signature (phase 2), typology_predictor, TR §5.5,
brisa P4/E2 figures.
