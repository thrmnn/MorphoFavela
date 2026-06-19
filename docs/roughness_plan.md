# Roughness-Estimation Track — Plan & Literature Brief

*Draft 2026-06-19, branch `track/roughness` (off main, which now holds the
morpho-signature work). Reviewable; nothing built yet. Same plan + append-only
decision-log discipline as the morpho-signature track.*

## Thesis

Aerodynamic roughness — roughness length **z0** and zero-plane displacement
**zd** — is the boundary condition that sets the wind a CFD run sees. For
favelas it is both **important** (it drives ABL inlet profiles and any
distributed-drag canopy parameterization) and **genuinely unsolved**: the SOTA
sweep found **no published z0/zd estimate for any favela, slum, or informal
settlement, by any method**. Every morphometric formula (Lettau → Macdonald →
Kanda → Millward-Hopkins) was calibrated on cube/obstacle arrays at
**λp ≈ 0.05–0.5**; favela fabric runs **λp ≈ 0.5–0.7**, with high height
variability and on **slopes** — all outside the calibration envelope.

We are unusually well-placed: the WS-0 feature substrate **already holds every
input** these methods need (λp, λf in 8 directions + mean/max, H_mean, σH; H_max
derivable from footprints), and we have sparse CFD patches that can **anchor**
the estimate (the same surrogate logic as the WS-B prioritization index). The
open contribution is *CFD-anchored morphometric roughness for steep, dense,
height-heterogeneous informal fabric*, on the two axes (λp > 0.5 and slope) that
no prior work touches.

## What we already have (maps straight onto the methods)

Per 10 m cell (from `features_grid.parquet`): **λp, λf_{N..NW} + mean/max,
H_mean, σH**; footprints+heights → **H_max**, building areas. That is enough to
apply *every* method below in full — including the height-heterogeneity ones,
because σH is already a fabric feature (it is also a morphotype-signature axis).

## Method lineage + equations (κ = 0.40 throughout)

**Lettau (1969)** — z0 only, strawman: `z0 = 0.5·H·λf`. Linear, no peak,
overpredicts above λf≈0.2–0.3.

**Macdonald, Griffiths & Hall (1998)** — drag-based baseline [inputs λp, λf, H]:
- `zd/H = 1 + α^(−λp)·(λp − 1)`
- `z0/H = (1 − zd/H)·exp{ −[ ½·β·(Cd/κ²)·(1 − zd/H)·λf ]^(−½) }`
- staggered α=4.43, β=1.0; square α=3.59, β=0.55; Cd=1.2. Calibrated λp≈0.05–0.44.

**Raupach (1994)** drag-partition [λf, H] — widest λ validity, best dense-canopy
physics; produces the z0 peak via a `u*/U ≤ 0.3` cap (c_d1=7.5, c_s=0.003,
c_r=0.3, ψh=0.193; urban re-tune c_r≈0.35).

**Kanda et al. (2013)** — σH-aware, **PRIMARY** [λp, λf (via z0_Mac), H_mean, σH,
H_max — we have all]:
- `zd = [ c0·X² + (a0·λp^b0 − c0)·X ]·H_max`, `X = (σH + H_mean)/H_max`, 0≤X≤1;
  a0=1.29, b0=0.36, c0=−0.17.
- `z0 = [ b1·Y² + c1·Y + a1 ]·z0_Mac`, `Y = λp·(σH/H_mean)`, Y≥0;
  a1=0.71, b1=20.21, c1=−0.77.
- The **super-linear b1=20.21** term in Y is exactly where Kanda departs *upward*
  from Macdonald for large σH/H_mean — the favela regime. Fit on building-resolving
  LES over real Tokyo/Nagoya squares + idealized arrays.

**Millward-Hopkins et al. (2011)** — σH-aware **cross-check** [λp, λf, H_mean, σH;
no H_max]: a uniform-height drag-partition core plus an explicit additive
`(σH/H_mean)` term for both zd and z0 (different functional form from Kanda —
agreement is a strong signal, divergence flags out-of-envelope fabric).

**Grimmond & Oke (1999)** — sanity bounds: zd≈0.7H, z0≈0.1H; benchmark sensitivity
study; documents wind-direction dependence.

## The 4 methods to implement & compare

| method | role | known failure on favela fabric |
|--------|------|--------------------------------|
| **Kanda 2013** | primary (only one using full σH+H_max) | X capped at 1 (saturates); fit where σH/H_mean is *lower* than favela extremes; λp>0.5 outside envelope → flag as extrapolation |
| **Millward-Hopkins 2011** | independent cross-check (additive vs Kanda's multiplicative) | high-λp ceiling; additive term misbehaves when σH/H_mean *and* λf both large |
| **Macdonald 1998** | H_mean-only baseline (Kanda−Macdonald = the "σH premium") | systematically *under*-predicts z0 for heterogeneous fabric (Zaki: up to ~2×); λp>0.44 extrapolation |
| **CFD drag-centroid (Jackson 1981)** | the anchor / validator | κ≠0.40 over very rough canopies makes fitted z0 ambiguous; patch-effective ≠ tower-averaged |

Compute **per cell** and as a **directional roughness rose** z0(θ) from the
8-direction λf — favela packing is anisotropic; don't collapse to mean too early.

## Where cube-array calibration breaks (flag explicitly in outputs)

1. **λp > 0.5** — every classical method *and* Kanda/MHN calibrated below ~0.5.
   Kent et al. (2017a) already find morphometric z0 is only ~31–43% of anemometric
   z0 (zd 10–42%) *before* favela extremes.
2. **The rougher-vs-smoother tension (the headline open question).** Cube curves
   peak z0 near λp≈0.2 then decline (skimming). But strong height randomness
   (Zaki 2011, Hagishima 2009, Cheng & Castro) *suppresses* skimming — tall
   outliers keep extracting momentum — so variable-height favela fabric may be
   *substantially rougher* than a same-mean uniform array. λp>0.5 skimming argues
   *smoother*; height randomness argues *rougher*. **Nobody has measured which
   wins** — our CFD anchors can.
3. **Anisotropy** — no street hierarchy → strong z0/zd direction dependence (use
   the 8-dir λf).
4. **Abutting party walls** — contiguous walls remove the gaps the drag-partition
   assumes; effective Cd ill-defined.
5. **Steep terrain — entirely unmodeled** by any roughness method. Open axis.

## CFD integration

- **Extraction (anchor/validate):** Jackson drag-centroid `zd = Σ zᵢDᵢ / Σ Dᵢ`;
  `u*² = τ_total/ρ`; fit the **double-averaged** ⟨Ū⟩(z) above canopy to
  `U=(u*/κ)·ln((z−zd)/z0)` with zd pinned, z0 the only free parameter; report if κ
  had to move off 0.40 (Leonardi & Castro 2010).
- **Consumption — two routes:** (i) **wall-function** equivalent sand-grain
  `ks = 9.793·z0/Cs` (code-specific; Cs≈0.5; mind the Blocken et al. 2007
  near-wall-cell consistency trap) + Richards & Hoxey (1993) ABL inlet
  `U=(u*/κ)ln((z+z0)/z0)`, `ε=u*³/(κ(z+z0))`; (ii) **distributed-drag** momentum
  sink `S = −ρ·Cd·a(z)·|U|·U` with `a(z)=dλf/dz` per wind sector — the direct
  consumer of our **per-direction λf** profile, an alternative to resolving every
  building.

## Validation & the open contribution

No favela z0/zd exists to validate against. So: (a) **cross-method spread**
(Kanda vs MHN vs Macdonald) as the morphometric uncertainty band; (b) **CFD
drag-centroid** on the sparse patches as local ground truth → recalibrate the
Kanda coefficients for favela fabric (the Kanda *workflow* — LES → refit a
morphometric formula — applied to favelas); (c) publish per-cell **extrapolation
flags** (X→1, λp>0.5, abutting-wall fraction, slope) so estimates are never
mistaken for in-envelope. Anemometric ground truth (Roth 2000 factor-~2 scatter;
Giometto 2016 "uniform city errs ~200%") frames the achievable accuracy.

## Workstreams & sequencing

- **R-0 · Roughness inputs** — derive per-cell H_max + abutting-wall fraction +
  slope from footprints; per-direction λf(z) slab profiles. Mostly already in the
  WS-0 substrate; thin extension (`src/morphometry/roughness_inputs.py`).
- **R-A · Morphometric z0/zd** — implement Lettau/Macdonald/Kanda/MHN as pure
  functions (`src/morphometry/roughness.py`), per-cell estimates + cross-method
  spread + extrapolation flags. Tests on cube-array values from the papers.
- **R-B · Directional roughness rose** — z0(θ)/zd(θ) from 8-dir λf; per-site rose
  figures in the hub.
- **R-C · CFD-anchored calibration** — Jackson drag-centroid on patches →
  patch-effective z0/zd → validate/recalibrate Kanda for favela fabric.
- **R-D · CFD consumption** — emit per-patch z0 (+ optional a(z) drag profiles)
  to the CFD repo contract; resolve the rougher-vs-smoother question with the
  anchors. Figures + decision log throughout; surface in the project hub.

## Key references (verify caveated constants before publishing)

- Macdonald, Griffiths & Hall 1998, *Atmos. Environ.* 32(11) — baseline equations.
- **Kanda et al. 2013, *Boundary-Layer Meteorol.* 148(2)** — σH-aware, primary.
- Millward-Hopkins et al. 2011, *BLM* 141 — height-distribution cross-check.
- **Kent et al. 2017a, *BLM* 164 (open access, PMC6979542)** — method
  intercomparison + every equation in common notation; cite for airtight constants.
- Raupach 1992/1994, *BLM* 60 / 71 — drag-partition, dense-canopy physics.
- Grimmond & Oke 1999, *J. Appl. Meteorol.* 38(9) — sensitivity benchmark, rules.
- Zaki et al. 2011 *BLM* 138; Hagishima et al. 2009 *BLM* 132 — height-randomness
  re-calibrations (cube constants under-predict z0 up to ~2×).
- Jackson 1981, *JFM* 111 — zd as drag centroid (CFD extraction).
- Blocken, Stathopoulos & Carmeliet 2007, *Atmos. Environ.* 41(2) — ks–z0 / wall
  function trap. Richards & Hoxey 1993, *JWEIA* 46–47 — ABL inlet.
- Coceal & Belcher 2004 *QJRMS* 130; Santiago & Martilli 2010 *BLM* 137 —
  distributed-drag / canopy (Lc = (1−λp)H/(Cd·λf)).
- Duan & Takemi 2021 *JAMC* 60; CFD-to-ML 2024 *Sustain. Cities Soc.* — proof a
  sparse-CFD-trained morphometric surrogate works.
- Stewart & Oke 2012, *BAMS* 93 — LCZ roughness classes (the only, and poor,
  "favela" proxy).

*Verification caveats from the sweep:* Kanda's LES solver is almost certainly PALM
(Gryschka/Raasch) but unconfirmed from open text; Raupach ψh and the Kent
equation constants were cross-checked against the open-access Kent 2017a reprints,
not the paywalled originals — pull a library copy before quoting exact values in
the paper.
