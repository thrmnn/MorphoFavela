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

## 3. Ventilation-axis methodology change (round-3 §1) — the bigger piece

Build a **multi-variable geometric ventilation-potential index** =
f(λf_dissolved, canyon aspect H:W, cluster openness/porosity), **excluding SVF**
(keep the solar and ventilation axes independent). Continuous underneath,
displayed as **ordinal tertiles + provisional hatching**. Recast the sparse
"ventilation-only" state as a morphological finding ("in fabric this uniformly
enclosed, ventilation deprivation almost always co-occurs with solar
deprivation"). ⟶DECIDE: adopt the index now (new analysis + Methods rewrite) vs
keep the λf-alone interim proxy with the round-3 placeholder label.

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

## 5. Still blocked / PI-level
- ⛔ Ray-caster vs Radiance/SOLWEIG — not installed locally.
- ⟶DECIDE (round-2): full lexical rename of taxonomy states to "Below … floor
  (owed)" across the manuscript body; pull SVF–sun decoupling into Fig 2 vs ED.

## Suggested order
Review prototype (you) → §2.1 validate 5 sites → §2.2 core integration → §3 index
(if adopted) → §4 figure regen against the new index. §4 cross-cutting gate can
proceed in parallel since it's mechanical.
