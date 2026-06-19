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

**Pending:** R-B map the rose spatially; R-C CFD drag-centroid anchor on patches to
resolve the dense-site collapse + recalibrate Kanda for favela fabric; R-D emit
per-patch z0 to CFD (decouple the two z0 roles — morphometric z0 → inlet/k_eq,
ground z0 small inside the resolved patch).
