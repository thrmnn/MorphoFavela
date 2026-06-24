# Typology as a Predictor of Environmental Failure — Plan

*Draft 2026-06-19 (from an expert brainstorm). A sub-study: use the discrete
morphological typology (cell morphotype + block morphotope, already learned from
geometry alone) as a calibrated, transferable predictor of environmental
deprivation. Nothing built yet; this is the plan for review.*

## Thesis

The discrete typology is a **calibrated, transferable** predictor of environmental
deprivation: given only a cell's type, state the expected probability of WHO-2 h
winter-sun failure with an honest, out-of-site uncertainty band. The discrete type
does **not** beat the continuous fabric vector on accuracy — it nearly matches it,
and *that near-equality is the contribution*: a 6-symbol code carries most of the
signal an 11-D vector does, while being communicable, threshold-interpretable, and
transferable to any favela mapped with morphology only.

Two qualifiers that keep this from being "density predicts deprivation, again":

1. **Semi-independent outcome.** The headline target is **WHO-2 h winter-sun
   failure**, ray-cast on the real 3-D DSM and *held out of the clustering*. SVF is
   partly geometric → demoted to a circularity-flagged secondary target, never the
   headline.
2. **Regime, not monotone correlation.** The typology earns its keep only if failure
   is *non-linear in type* (a jump at the flat/steep or T3→T4 boundary, a plateau
   across T0–T2, saturation at T5). If the per-type curve is a straight line, the
   discretisation adds nothing — and we report that honestly.

## Analysis plan (one row per built 10 m cell; 5 CFD sites + 3 calibration held aside)

- **Step 0 — autocorrelation scale.** Variogram range of the failure residual per
  site sets the spatial-block size for all CV/CIs (random CV gives 5–20× false
  confidence — Roberts 2017, Ploton 2020).
- **Step 1 — per-type failure tables** with **block-bootstrap CIs** (resample
  spatial blocks/sites, not cells). The "tell me the type → failure rate" lookup.
- **Step 2 — parsimony test (core):** (A) type one-hot, (B) continuous vector,
  (C) both; penalised logistic *and* gradient-boosted trees; metric ΔAUC-PR + Brier.
  Thesis lives in (B)−(A): small gap ⇒ type keeps the signal. Bin the continuous
  model's predicted prob by type to show the boundaries coincide with regime jumps.
- **Step 3 — leave-one-site-out transfer (the real test):** train on 4 favelas,
  predict the 5th from type alone; per-fold PR + calibration + per-type rate shift.
- **Step 4 — calibration curves** (reliability diagram, ECE; isotonic recal on
  training sites only). A prioritisation proxy must be calibrated, not just rank-correct.
- **Step 5 — PR/ROC** for WHO-2 h, LOSO-pooled, lead with PR + the prevalence baseline.
- **Step 6 — imbalance + circularity audit:** class weights (no SMOTE); refit with
  SVF in vs out to quantify the non-geometric residual predictive power.

## Cross-site deepening

- **C1 — variance decomposition:** 3-level logistic mixed model
  `who2h_fail ~ morphotype + (1|site) + (1|site:morphotype)`; VPC splits failure
  variance into between-type (the proxy's signal) / between-site (topography the
  type misses) / **site×type interaction** (= transfer breakdown). The site×type
  term being large is the most likely real result and is itself a finding.
- **C2 — mediation:** does the type effect survive conditioning on λp, H, slope? A
  residual = the configuration/regime info the scalars miss (the type's reason to exist).
- **C3 — orientation/latitude moderation:** sites differ in aspect; pre-register
  aspect/latitude as moderators of the type→sun mapping (the physical reason transfer
  might fail), not a post-hoc discovery.
- **C4 — cell vs morphotope level:** block-scale should transfer better, discriminate
  worse; reporting both resolutions addresses the ecological-fallacy risk.

## Transfer / payoff

A **morphology-only environmental-risk lookup** keyed on (morphotype × morphotope),
each cell carrying `{p̂_failure, CI, calibration flag}`, validated LOSO on the 5 CFD
sites, then **applied blind to the 3 calibration favelas** (morphology only, never in
training) → a prioritisation map: which areas to survey/simulate first. Relation to
**Local Climate Zones** (Stewart & Oke 2012): adopt the typology-as-climate-proxy
epistemics, critique the resolution — LCZ collapses all favela fabric into 1–2
classes; **morphotypes are a favela-resolution, outcome-specific LCZ for solar/
ventilation deprivation.**

## The 3 sharpest risks + mitigations

- **R1 Circularity (the killer):** SVF/deep-canyon are geometric functions of the
  cluster features. → WHO-2 h ray-cast sun is the *sole* headline target; SVF demoted +
  circularity-audited (Step 6).
- **R2 Ecological fallacy:** a 10 m cell's modelled failure ≠ a resident's lived
  deprivation. → report at cell *and* morphotope resolution; frame as survey/simulation
  prioritisation, never household-level.
- **R3 Transfer breakdown + pending physics:** site×type may be large; ventilation is
  CFD-pending; morphometric roughness is invalid at favela density. → scope the headline
  to **solar (validated, transferable)**; label ventilation *projected, pending CFD*;
  never let a roughness proxy stand in for ventilation.

## Figures that carry the argument

1. **The lookup panel (money figure):** per-type WHO-2 h failure rate (T0→T5 + the 5
   morphotopes), block-bootstrap CIs, with per-site points overlaid — regime curve +
   transfer-vs-moderation in one glance.
2. **LOSO transfer + calibration twin-panel.**
3. **Variance-partition bar** (between-type / site / interaction) + the parsimony Δ inset.
4. **The blind risk map** — a calibration favela the model never saw, morphology in →
   prioritised environmental-risk out.

## Implementation

A new analysis script `scripts/analyze_typology_predictor.py` consuming the existing
per-cell tables (`features_grid.parquet` morphotype/morphotope + the held-out WHO-2 h
sun flag) — no new upstream pipeline. Needs `statsmodels`/`scikit-learn` (have) +
possibly `glmmTMB`-equivalent for the mixed model (`statsmodels` MixedLM or a Bayesian
`bambi`; decide at build). Verify the search-abstract references (Quan & Li; the
facade-solar ML; the LCZ-transfer paper) before they enter a bibliography.
