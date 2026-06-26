# Expert-council roadmap & Track C re-scope (2026-06-25)

Output of a 5-expert council (urban climatologist · CFD/wind · morphometrics/GIS ·
ML/spatial-stats · journal editor) + synthesis. This is the standing plan for
continued autonomous work. Engineering contract below is **non-negotiable**.

## Non-negotiable engineering contract (all five personas agreed)

1. **Additive, suffixed columns only.** Never overwrite the locked canonical λf
   (`outputs/brisa_ventilation_fix/lambda_f_canonical.json`, test-pinned). It must
   reproduce bit-for-bit after any change.
2. **Every downstream shift is a documented A/B sensitivity, never a silent
   re-baseline** (regime shares, morphotypes, predictor).
3. **Spatially-blocked CV** for any predictor claim (Moran's I 0.33–0.70 → naive
   LOSO is optimistic). Report shifts inside the block-bootstrap CI as "immaterial."
4. **Nothing here rescues the roughness z0** — terrain-following λf is still ≫0.5,
   so the 53–75 % out-of-envelope invalidity stands as its own contribution.

## Prioritized roadmap (rank = scientific_value × local-feasibility)

1. **[deepen/M] Spatial-CV hardening of the predictor FLIP** — ✅ DONE (commit fc50a72).
   Site-block-bootstrap 95% CI: type 0.55 [0.48,0.77], vector 0.86 [0.76,0.93] —
   **CI-overlapping**, so the flip is a direction not a separated gap (5 sites → wide
   bands). VIF low (1.1–3.0). `spatial_cv.json` + CI error bars on fig_parsimony. The
   council's predicted "gap may shrink toward 0.7" is borne out.
2. **[deepen/S] Calibrate the blind risk map** — ✅ DONE (commit 3bfcf86). Pooled
   AUC-PR 0.76 / Brier 0.20 on the 3 held-out favelas (never in training); site-variable
   (jacarezinho 0.82 but 62% extrapolating; juramento 0.39 low-prevalence). Transfers as
   a stratifier with extrapolation flags. `typology_blind_validation.png`.
3. **[deepen/M] Tessellation + shape + grain descriptors into the morphotypes** —
   ✅ DONE (S1, commit d04c317). Additive `morphotype_shape` column; VIF-screened (dropped
   shape_index_mean 10.45); **shape is a separable axis, not density re-expressed** (cell
   LOSO-ARI 0.181, morphotope 0.436); canonical morphotype/morphotope/λf bit-for-bit.
   Decision log D21–D23. **ALL 7 roadmap items + the 6-stream repo-audit plan now complete.**
4. **[deepen/S] Directional λf × wind-rose → effective wind-EXPOSURE scalar** —
   Σ_θ freq(θ)·λf_θ per cell from the unused 8-sector λf + `wind_rose.json`. Frame as
   geometric exposure *tendency*, NOT adequacy (τ gated; z0 is 180°-symmetric → frequency
   only, never channelling). Verify sector-binning convention matches first.
5. **[deepen/M] Track C re-scoped** (see below) — σH arm dropped; λf-tilt arm gated.
6. **[deepen/S] Lateral-connectivity → 2D ventilation-susceptibility regime map** —
   ✅ DONE (commit 782507b). Bivariate regime × depth, worst = skimming×deep; pooled
   41.8% doubly constrained (RdP 60%). `ventilation_susceptibility.png`.
7. **[explore/S] MAUP sensitivity** — σH/λf + regime shares at 10 m vs 20 m (both grids
   on disk). Confirmatory rigor a methods reviewer will demand. ⏳ remaining.

## Status (2026-06-25): #1 #2 #4 #5(C) #6 DONE · remaining #3 (morphotype re-fit, M) · #7 (MAUP, S)
#3 is the next substantial unit — it touches the canonical morphotype fit, so it needs
the additive-column + A/B + LOSO-ARI contract; best as a focused effort.

## Track C — ✅ RESOLVED as a scoping null (2026-06-25, commit 9923a1c)

Steps 0–2 done: σH arm disproven (corr(slope,σH)=−0.12, already terrain-following);
doc corrected; λf terrain-step gate CLEARED (upper-bound median below 25° = 2.2% < 5%)
→ no Step 3, canonical λf untouched. Evidence `outputs/comparative/roughness/
slope_confound.json`. Full original plan retained below for the record.

## Track C — RE-SCOPED (chair sided with the ML dissent)

**Premise partly false (empirically confirmed):** grid σH is built on `altura`
(`grid.py:114`, terrain-following) and corr(slope, σH) is NEGATIVE on steep sites
(vidigal −0.08, rocinha −0.13, alemão −0.16, maré −0.02; rdp +0.09) — the wrong sign
for "slope inflates σH." So the σH recompute is a near-no-op → **dropped**. Only the
2D-plan λf face-projection on slopes is a real candidate artefact, and it may be
second-order (cos 25° = 0.91) → **hard magnitude gate before building anything.**

- **STEP 0** — reproduce the per-site corr(slope, σH) / corr(slope, λf) table; commit
  as a negative/scoping result. (Council pre-ran it; confirm + persist.)
- **STEP 1** — correct `docs/roughness_decisions.md`: σH is already terrain-following;
  the z0-rises-on-slope pattern is NOT a σH datum artefact; residual candidate is λf
  projection only, magnitude TBD.
- **STEP 2** — scratchpad diagnostic: per-cell cos(slope) wall-tilt + along-wind
  terrain-step Δz = slope·span·cos(aspect−θ); report (λf_tilt−λf)/λf by slope bin.
  **GATE: if median correction <25° is <~5%, STOP and report the null as the finding.**
- **STEP 3** (only if gate clears) — additive `lambda_f_{N..NW}_tf` + `lambda_f_mean_tf`;
  reuse the exact `_projected_width` bearing convention (CW-winding gotcha); canonical
  λf untouched; tests staged same commit (flat-site identity + synthetic-ramp invariance).
- **STEP 4** — regime A/B sensitivity on λf_tf vs canonical 65/30/5, as a TR appendix.

**Validation:** flat-site null (λf_tf==λf at slope≈0); synthetic-ramp invariance (λf_tf
invariant to a uniform-gradient DTM, flat-datum control inflates); physical floor
(0 ≤ tilt face ≤ absolute face); partial-correlation honesty (report the likely null —
don't engineer toward residual-slope==0); reversibility guard (lambda_f_canonical.json
reproduces).

## Dissent on record
4/5 endorsed Track C as greenlit; the ML/spatial-stats expert dissented with code+data
that the σH premise is false. Chair adopted the dissent for the σH arm, kept λf gated.
Secondary (non-blocking): climatologist's g_a = u*/U conductance bridge inherits the z0
out-of-envelope invalidity → left off (defensible only as a regime-conditioned band).
