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

## 4b. Morphotype-cascade stability (measured 2026-06-24) — DECISION NEEDED

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
