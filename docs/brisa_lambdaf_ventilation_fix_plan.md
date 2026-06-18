# BRISA+ — Ventilation geometric-proxy fix plan (λf repair + H/W proxy + SVF link)

**Status:** OPEN — handed off from the brisa_paper conversation 2026-06-02.
**Owner:** separate onboarded agent (this repo, MorphoFavela). Do NOT edit `brisa_paper/artifacts/latex/main.tex` from here — produce corrected grids + stats; the paper integrates them when the user signals.
**Why this exists:** the manuscript's geometric ventilation pre-screen (λf > 0.35) was found to be broken three independent ways. The paper has been demoted to treat it as an uncalibrated interim proxy. This plan produces a *correct* geometric ventilation proxy so the paper can stand on a sound interim axis until the CFD-ACH campaign lands.

---

## 1. The problem (forensic findings, verified 2026-06-02)

Three stacked flaws in the **grid** λf product:

**(A) Computation bug — no cell clipping.**
- `src/urban_morphology.py:compute_frontal_area_ratio` (~L262–297) and `src/morphometry/indicators.py:compute_lambda_f_directional` (~L36–62).
- Buildings are attributed to a cell by `gpd.sjoin(predicate="intersects")`, then each building contributes its **entire** minimum-rotated-rectangle frontal area (full projected width × full height) to the cell — divided by the fixed 10 m cell area (`zone_area = 100 m²`, `src/morphometry/grid.py:compute_grid_morphometrics`).
- **No clipping**, unlike `compute_bcr` (~L136) which correctly clips via `row.geometry.intersection(zone_geom)`.
- Result: grid `lambda_f_mean` runs **0–54** (median 2.5–5.6 across sites) vs canonical Grimmond & Oke 0–1.5. It correlates **r ≈ 0.82 with raw `building_count`** on Vidigal — i.e. it is effectively a building-count proxy. (λp, which IS clipped, correctly stays 0–1.)

**(B) Wrong threshold from the wrong variable.**
- 0.35 is **not** a λf skimming-flow onset. It is the Oke (1988) **H/W aspect-ratio** isolated→wake-interference boundary (H/W ≈ 0.35) — a different variable *and* one regime below skimming.
- Skimming onset is canonically **H/W > 0.65** (Oke 1988), ≈ **λf 0.4–0.5** in the regular-array equivalence (Macdonald et al. 1998).
- Grimmond & Oke (1999), currently cited, parameterizes **bulk aerodynamic roughness (z₀, z_d)** — not a per-cell pedestrian-ventilation criterion. The z₀/H peak (λf-crit) is ~0.11–0.3, not 0.35.

**(C) Saturation is the artifact of A+B.**
- Because values are inflated 7–16×, 0.35 sits below the 5th percentile everywhere → 95–99% of built cells "exceed" it (even >1.5 captures 70–93%). Non-discriminating by construction.

**Important nuance:** the **per-patch** λf used in CFD sampling (`outputs/<site>/cfd_analysis/per_patch_indicators.csv`, read at `scripts/run_diagnostic_models.py:~346`) is computed on large domains and comes out **canonically correct (0.37–2.70)**. Only the **grid** product (`outputs/<site>/morphometrics/grid/grid_metrics.csv`, read at `run_diagnostic_models.py:~102`) is broken. Keep the two products distinct.

---

## 2. Per-site grid λf distribution (built cells, current/broken product)

| Site | n | median | p1 | p5 | max | %>0.35 | %>1.5 |
|---|---|---|---|---|---|---|---|
| Vidigal | 2756 | 2.78 | 0.22 | 0.48 | 13.1 | 97.4 | 75.5 |
| Rocinha | 8031 | 4.36 | 0.11 | 0.53 | 22.3 | 97.0 | 85.0 |
| Alemão | 17768 | 2.49 | 0.18 | 0.35 | 23.0 | 95.2 | 69.8 |
| Rio das Pedras | 6605 | 5.62 | 0.42 | 1.17 | 24.2 | 99.4 | 92.7 |
| Maré | 29229 | 4.49 | 0.23 | 0.67 | 54.6 | 98.1 | 86.7 |

---

## 3. The plan (three prongs, run in PARALLEL)

### Prong A — Repair grid λf (cell-clipped)
- Clip each building footprint to the cell (as `compute_bcr` does) **before** computing projected frontal area, OR use a physically meaningful cross-wind cell footprint as the denominator.
- Target: grid `lambda_f_mean` lands in the canonical ~0–1.5 band; correlation with `building_count` drops sharply.
- Re-pin the threshold to a **defensible** value: λf ≈ 0.4–0.5 (WIF→SF transition, Macdonald et al. 1998), cited correctly.
- **Acceptance:** new per-site distribution where the chosen threshold is a genuine discriminator (not >95% exceedance); recompute the four-state taxonomy shares; write `outputs/<site>/morphometrics/grid/grid_metrics_v2.csv` + a short stats JSON. Add a regression test asserting clipped λf ≤ unclipped λf and median in band.

### Prong B — H/W canyon aspect-ratio proxy (the canonically-correct skimming variable)
- Derive street-canyon aspect ratio H/W (building height ÷ street/void width) along the pedestrian network. Flag skimming-likely at **H/W > 0.65** (Oke 1988).
- This is the *physically correct* skimming-flow flag; it can stand as the primary geometric ventilation pre-screen if it discriminates.
- **Acceptance:** per-site H/W distribution + skimming-flag share; sanity-check against the corrected λf (Prong A) and against SVF (Prong C). Output `outputs/<site>/morphometrics/canyon/hw_streets.gpkg`.

### Prong C — SVF–ventilation link (the robust, validated anchor)
- The diagnostic models already use **street-level** SVF (`outputs/<site>/morphometrics/svf/svf_streets_solar.gpkg`, `run_diagnostic_models.py:87`), which is UMEP-validated (`scripts/validate_svf_against_umep.py`) and more robust than grid SVF (grid SVF includes degenerate rooftop/interior cells). Standardize the ventilation-openness story on **street SVF**.
- Establish/quantify the SVF↔ventilation relationship (literature anchor Yang et al. 2022: SVF↔wind-velocity ratio). When CFD-ACH lands, the SVF–ACH changepoint (paper Fig 5D / §2.7) is the test.
- **Acceptance:** a street-level SVF-based ventilation-potential map per site + the SVF distribution stats; document why street SVF > grid SVF.

---

## 4. How results flow back to the paper
- Produce corrected grids + stats JSONs + figures in this repo. Do **not** edit the manuscript.
- When a prong is validated, the user signals the brisa_paper session to pull the numbers (the paper currently carries honest interim/provisional language that these results will replace).
- The CFD-ACH campaign is running on its own separate track; it supplies the *calibrated* ventilation axis and supersedes all geometric proxies. These three prongs are the **interim** sound proxy, not a replacement for CFD.

## 5. Citations to get right
- Oke (1988) "Street design and urban canopy layer climate", Energy & Buildings 11:103–113 — H/W skimming regimes (NOT yet in brisa_bib; add).
- Macdonald, Griffiths & Hall (1998), Atmos. Environ. 32:1857–1864 — λf WIF→SF ≈ 0.4–0.5 (add).
- Grimmond & Oke (1999) — keep, but cite ONLY for bulk z₀/z_d roughness, not a per-cell threshold.
- Yang et al. (2022) — SVF↔wind velocity ratio (already in brisa_bib).
