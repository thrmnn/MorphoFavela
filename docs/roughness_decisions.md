# Roughness-Estimation Track — Decision Log

Append-only record of choices + findings. Companion to `docs/roughness_plan.md`.

## R-A — Per-cell morphometric roughness

**RA-1 · UMEP is the engine, Kanda the primary method.**
`src/morphometry/roughness.py` wraps the vendored UMEP `RoughnessCalc(method, zH,
fai, pai, zMax, zSdev)` (Kent & Grimmond) — methods Kan/Mho/Mac/Rau — rather than
re-deriving formulas. Kanda 2013 is primary; Macdonald carried as the σH-free
baseline. A unit test pins UMEP's Kanda against the expected behaviour (incl.
zd > H_mean). NaN-guarded; vectorized over cells.

**RA-2 · H_max per cell from building heights.**
Derived by spatial-joining `buildings_with_morphology_metrics.gpkg` to the grid and
taking the max height per cell; fallback `H_mean + 2.5·σH` where the buildings layer
is absent. zMax is the one Kanda input not already in the WS-0 substrate.

**RA-3 · Outputs.** `features_roughness.parquet` per site: `z0_kan`, `zd_kan`, 8
directional `z0_kan_{N..NW}` (the roughness rose), `z0_{kan,mho,mac,rau}` +
`z0_method_spread` (the morphometric uncertainty band), `H_max`,
`zd_exceeds_Hmean`, and the extrapolation flags. Directional rose figure in the
gallery (`roughness_rose.png`).

**RA-4 · Findings (face-valid + the headline caveat).**
- **zd > H_mean in 70–93% of built cells** across sites — heterogeneous favela
  fabric pushes displacement above the *mean* height (tall outliers dominate drag),
  exactly as Kanda/Kent predict and the mentor flagged. Not a bug.
- **λp > 0.5 in 56–88% of cells** — most favela fabric is **outside the calibration
  envelope of every method** (all fit below ~0.5). Surfaced per cell via
  `flag_pai_over_envelope`, never silently.
- **Densest flat sites (jacarezinho, riodaspedras) → z0 collapses** (zd→H, skimming
  regime): z0_kan medians 0.0003 / 0.020 m vs 0.13–0.22 m for the steeper/looser
  sites. This is precisely the **open rougher-vs-smoother question** (λp>0.5
  skimming says *smoother*; height-randomness says *rougher*) — flagged as
  extrapolation, to be resolved by the CFD anchors (R-C), not trusted as-is.
- **Cross-method z0 spread ≈ 0.30–0.40 m** (median) — large; the morphometric
  uncertainty is real and reported, not hidden behind a single method.

## R-B — Spatial roughness map

**RB-1 · Per-cell z0 choropleth** (`roughness_map.png`, cividis — no clash with the
morphotype Okabe–Ito or the priority YlOrBr), shared robust scale (pooled p95),
NULL grey, dissolved boundary + scale bar, constant ground-scale small-multiple.
Confirms the R-A finding spatially: **dense cores collapse to low z0** (skimming),
**looser/edge cells are rougher** — the gradient runs opposite to intuition because
λp>0.5 skimming dominates the interiors. Title carries the out-of-envelope caveat.
Cell-level for now (dissolve is the polish step).

**RA-5 · Cross-method disagreement is the headline uncertainty (figure
`roughness_methods.png`).** Plotting median z0 by Macdonald/Raupach/Millward-
Hopkins/Kanda per site: in the favela λp>0.5 regime the four **span up to ~20×**.
The naive expectation — σH-aware Kanda/MHN lift z0 above the height-blind Macdonald
("heterogeneity premium") — **does not hold in the dense regime**: Kanda's
quadratic-in-Y correction can drop z0 *below* Macdonald in the skimming limit
(Maré, Rio das Pedras), while Raupach/MHN run high. So the honest figure is
*disagreement*, not a clean premium — and *which method is right is unknown without
CFD*. This is the validation gap quantified, and it directly motivates R-C. (Caught
on review: the first cut mis-titled this "the σH premium" before the data was read.)

**RB-2 · Two findings-driven maps (track `track/viz`).** (i) `roughness_zd_ratio.png`
— diverging map centred at zd/H_mean = 1 (RdBu, TwoSlopeNorm): red where displacement
exceeds mean height, the spatial face of the 70–93% finding. (ii) `roughness_slope.png`
— median z0 binned by terrain slope (≥30 cells/bin, capped at 35° where data thins).
z0 *rises* on steep slopes, but this is a **confound, not a clean finding**.
**Correction (2026-06-25, Track C Step 0):** the earlier claim that a *flat-datum*
σH/λf "absorbs the hillside" is **wrong** — grid σH and λf are built on `altura`
(height above each building's own base; `grid.py` sets `height = altura`) and are
already terrain-following. Empirically corr(slope, σH) is small and **negative** on
the steep sites (pooled −0.12; corr(slope, λf) −0.10 — see
`outputs/comparative/roughness/slope_confound.json`), the wrong sign for a
datum-inflation mechanism. The z0–slope pattern is a fabric/method co-variation, not
a morphometric datum artefact; the only residual geometric candidate is λf's 2-D
plan-view face projection on slopes, whose magnitude is gated in Track C Step 2.

## Expert council review (2026-06-19) — the correctness reckoning

Three experts (urban-climate/roughness · morphometrics · CFD) reviewed the track.
Convergent verdict and what changed:

**RC-1 · The per-cell morphometric z0/zd is NOT a valid measurement at favela
density.** It is physically INVALID in **53–75% of cells**: zd > H_max (displacement
above the tallest building — impossible; median λp in these cells = 1.0, Kanda X
saturated ~0.97) or z0 → 0 (skimming asymptote — "smoother than grass"). λf ≈ 1.7–2.1
and λp > 0.5 are 1–2 orders past every method's calibration (arrays top out ~λf 0.4,
λp 0.4); the drag-partition exponential saturates, so z0 is set by σH/H_max, not
fabric. **Reframe: model extrapolation, not measurement.** New `roughness_validity.png`
leads the gallery; `flag_zd_gt_Hmax` / `flag_z0_collapsed` / `roughness_physically_valid`
added per cell. H_max clipped ≥ H_mean (fixed a ~5% derivation mismatch — not the cause).

**RC-2 · The method-spread envelope IS the result.** With ~20× divergence and zero
favela validation, lead with the 4-method min–max band, not any single method:
"morphometry cannot constrain favela z0 to better than ~1.5 orders; CFD required."

**RC-3 · The 180° symmetry is a fundamental constraint, not a finding.** Frontal area
is direction-symmetric → z0(θ)=z0(θ+180°) for *all* morphometric methods. Morphometry
can never represent channelling / N-vs-S asymmetry; only CFD can. Documented as the
boundary of geometry-only roughness and the motivation for the campaign.

**RC-4 · The site-median rose cancels per-cell anisotropy (phase cancellation).**
Per-cell λf anisotropy is real (median ~0.37) but cell orientations vary, so the
per-sector median is near-isotropic — an artefact. Added `roughness_anisotropy.png`
(the honest directional statistic); rose radial axis zoomed + caveated; the z0 rose
is demoted (show λf(θ)/anisotropy instead). Patch-scale z0 should use momentum/flux
blending, not a median of cell z0.

**RC-5 · Cell z0 map is illustrative only** — z0 is a blended quantity over many
elements; a 10 m cell is smaller than one element wake. The 100 m patch is the
smallest defensible scale. Slope figure retitled as a confound (flat-datum λf/σH
absorbs the hillside); fix is terrain-following morphometry; slope stays out of the
inlet z0.

**RC-6 · CFD coupling hardened** — inlet from the upstream fetch (not the patch),
two-zone ground, z0 floor for λp>0.5, drag-integral (not Jackson) for steep patches,
empty-domain homogeneity + GCI gates. Full list in `src/cfd_integration/README.md`.

**RC-7 · Signature gaps the morphometrician flagged** (for the signature track, not
roughness): missing configuration (party-wall adjacency, street-network/beco width);
λp censored at 1 in T5 (dense end is λf-defined; 10 m may under-resolve the beco);
recurrence claim is stronger at a block-scale *morphotope* than per cell. Logged here
as cross-track; see [[morpho-signature-track]].

## RC-8 · Validity response — ADOPTED (user, 2026-06-19)

Decision on how to handle the per-cell roughness invalidity (RC-1):

- **A — now:** lead with the **method-spread envelope + validity flags**, not any
  single z0; frame absolute roughness as CFD-gated. The `roughness_validity.png`
  figure leads the gallery; §6.5 and the overview state the envelope is the result.
  *Adopted as the reporting position.*
- **B — CFD hand-off:** use the **patch-scale** morphometric z0(θ) (plausible band,
  flags carried) as the CFD inlet prior — never the per-cell value. `patch_roughness.csv`
  is the artifact; the contract is in `src/cfd_integration/README.md`.
- **C + D — research moves (planned, not now):** terrain-following morphometry (to
  strip the slope confound from σH/λf) and a sheltering/porosity correction
  (Millward-Hopkins-style, to cap drag as λf→high). Neither substitutes for the CFD
  anchor (R-C); both are post-review work.

## R-C / R-D — gated on real CFD

**RD-prep · Per-patch morphometric z0(θ) computed (the CFD-inlet hand-off).**
`scripts/build_patch_roughness.py` runs UMEP/Kanda per campaign patch from the
patch-scale morphometry in `patch_meta.json` (λp, H_mean, σH, H_max) + the
patch-aggregated λf(θ) over the cells inside the 100 m analysis disk. Output:
`outputs/{site}/sampling_cfd/campaign_sampling/patch_roughness.csv` (+ pooled
`outputs/cross_site/roughness/patch_roughness.csv`) — per-patch z0(θ)/zd(θ) to set
each patch's CFD **inlet ABL + k_eq**. *Outcome:* 119 patches, 118 with z0, **77
flagged λp>0.5**; per-site median z0 0.10–0.61 m. **Scale note:** patch-effective z0
(100 m disk) ≠ the cell-level value — the per-cell skimming collapse (R-A) averages
out at patch scale, and the patch scale is the CFD-inlet-relevant one. This is the
*morphometric* z0; the CFD-*extracted* z0 of R-C is the validator.

**RC-gate · The CFD drag-centroid anchor needs real OpenFOAM force/velocity fields**
and will NOT be run on the synthetic placeholders (`data/{site}/cfd_results/` holds
unmarked synthetic data). R-C (extract patch-effective z0/zd via Jackson + log-fit)
and R-D (wire per-patch z0 into the CFD contract) wait for the campaign.

**Pending:** R-C CFD drag-centroid anchor on patches to
resolve the dense-site collapse + recalibrate Kanda for favela fabric; R-D emit
per-patch z0 to CFD (decouple the two z0 roles — morphometric z0 → inlet/k_eq,
ground z0 small inside the resolved patch).
