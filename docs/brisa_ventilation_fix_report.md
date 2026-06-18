# BRISA+ ventilation-proxy fix — handoff report

**Date:** 2026-06-02
**Scope:** corrected products + stats for the three prongs of
`docs/brisa_lambdaf_ventilation_fix_plan.md`. The brisa_paper manuscript
integrates these numbers when the user signals; this repo deliberately
does **not** edit `main.tex`.

---

## 0. Forensic findings — re-verified

Re-ran the §1 forensic check on the original (broken) grid product:

- Vidigal built cells = 2 754; `lambda_f_mean` ran 0–13.1, median 2.78, p5 0.48
- `corr(lambda_f_mean, building_count) = 0.82` (Pearson)
- legacy threshold `λf > 0.35` fired on 97.5 % of built cells

Matches the plan-doc table verbatim. The unclipped accumulation in
`src/urban_morphology.py:compute_frontal_area_ratio` is the bug —
confirmed by source read and by the new regression test below.

---

## 1. Prong A — Repair grid λf (cell-clipped, **in-place overwrite**)  ✅ VALIDATED

**Code change.** `compute_frontal_area_ratio` now clips each building
footprint to the zone before computing projected width, mirroring
`compute_bcr`. New `clip_to_zone=True` default. Pass-through to
`compute_lambda_f_directional`. `_projected_width` guards against
degenerate Point/LineString intersections.

**Regression test.**
`tests/test_urban_morphology.py::test_clipping_caps_lambda_f_below_unclipped`
asserts clipped λf ≤ unclipped λf on a 30 m building straddling two 10 m
cells; passes. The full ratio test for square buildings still passes.

**Cleanup policy.** The repair script overwrites
`outputs/<site>/morphometrics/grid/grid_metrics.{csv,gpkg}` **in place**
with the clipped λf columns. The broken pre-overwrite distribution is
captured into `lambda_f_repair_stats.json` for the paper trail. No
`_v2` shadow files remain — downstream consumers (e.g.
`build_diagnostic_map.py`) automatically pick up the corrected values.

**Per-site distribution (clipped λf, built cells).**

| Site | n_built | p5 (broken→clipped) | p50 (broken→clipped) | max (broken→clipped) | r(λf, bldg_ct) broken→clipped |
|---|---|---|---|---|---|
| Vidigal | 2 754 | 0.48 → 0.17 | 2.78 → 1.31 | 13.1 → 5.5 | 0.82 → 0.76 |
| Rocinha | 8 029 | 0.53 → 0.20 | 4.36 → 2.19 | 22.3 → 20.0 | — → 0.66 |
| Alemão | 17 766 | 0.35 → 0.14 | 2.49 → 1.15 | 23.0 → 9.9 | — → 0.76 |
| Rio das Pedras | 6 605 | 1.17 → 0.33 | 5.62 → 2.69 | 24.2 → 10.9 | — → 0.70 |
| Maré | 29 227 | 0.67 → 0.17 | 4.49 → 1.60 | 54.6 → 10.7 | — → 0.67 |

**Coarse-grid aggregation** (mean of clipped 10 m λf within 50 m / 100 m
macro-cells — equivalent to clipped λf on macro-cells since clipped
facade area is conservatively partitioned):

| Site | 100 m p50 | 100 m %>0.40 | 100 m %>0.50 |
|---|---|---|---|
| Vidigal | 1.06 | 89.1 | 80.4 |
| Rocinha | 1.77 | 83.2 | 81.5 |
| Alemão | 1.03 | 83.8 | 78.0 |
| Rio das Pedras | 2.34 | 93.3 | 90.4 |
| Maré | 1.04 | 74.5 | 70.9 |

**Substantive finding.** Clipping cuts the median by 2–3× and the
λf↔building_count correlation from 0.82 to 0.66–0.76. But **even at 100 m
macro-cells**, Macdonald's λf ≈ 0.40 still fires on 74–93 % of built
macro-cells. Rio favelas are uniformly deeper into the skimming-flow
regime than the WIF→SF transition Macdonald characterised on regular
arrays. The corrected grid λf is a useful **relative pattern map**
(where is facade density highest within a site?) but is **not a useful
absolute ventilation pre-screen** at any reasonable cell size, because
the canonical thresholds were calibrated on far sparser geometries.

**Threshold pinning.** The defensible choices documented in
`lambda_f_repair_stats.json`:
- λf > 0.40 (Macdonald 1998 lower bound) — for cross-city comparison, but
  saturates within Rio favelas.
- λf > within-site p75 — relative discriminator for spatial mapping.
- λf > 1.5 (deep skimming) — actionable upper-tier flag.

Citations correctly attributed: Macdonald, Griffiths & Hall 1998 for
WIF→SF; Oke 1988 only for H/W (not λf); Grimmond & Oke 1999 retained
only for bulk z₀/z_d roughness (not per-cell).

**Outputs:**
- `outputs/<site>/morphometrics/grid/grid_metrics.{csv,gpkg}` ← **overwritten in place**
- `outputs/brisa_ventilation_fix/lambda_f_repair_stats.json`
- `tests/test_urban_morphology.py::test_clipping_caps_lambda_f_below_unclipped`

---

## 2. Prong B — H/W canyon aspect-ratio proxy  ✅ VALIDATED

For each ~10 m segment of the pedestrian network, the script samples a
midpoint, scans the perpendicular for the nearest flanking building on
each side (search radius 40 m), and records
`W = d_left + d_right`, `H = mean(h_left, h_right)`, `HW = H/W`.
Skimming flag: `H/W > 0.65` (Oke 1988).

| Site | n segments | H/W p5 | p50 | p95 | %skimming (>0.65) | %wake-interference (>0.35) |
|---|---|---|---|---|---|---|
| Vidigal | 971 | 0.23 | 1.36 | 7.15 | **76.1** | 89.6 |
| Rocinha | 5 204 | 0.15 | 2.12 | 11.98 | **76.7** | 85.6 |
| Alemão | 9 552 | 0.13 | 0.76 | 4.99 | **54.8** | 71.8 |
| Rio das Pedras | 2 946 | 0.19 | 1.91 | 9.93 | **77.5** | 85.5 |
| Maré | 9 906 | 0.11 | 0.81 | 4.94 | **56.5** | 73.3 |

**This is the physically-correct skimming-flow variable** and it
**discriminates cleanly**: Vidigal / Rocinha / Rio das Pedras sit at
75–78 % skimming (steep, dense), Alemão / Maré at 55–57 % (flatter,
wider streets). Differs by 22 pp across sites — i.e. a real signal,
unlike the broken λf which saturated everywhere.

**Outputs:**
- `outputs/<site>/morphometrics/canyon/hw_streets.gpkg` (5 files; one
  point per street segment midpoint with W, H, HW, d_left, d_right,
  h_left, h_right).
- `outputs/brisa_ventilation_fix/hw_streets_stats.json`

---

## 3. Prong C — Street-level SVF as ventilation-openness anchor  ✅ VALIDATED

Standardised on `outputs/<site>/morphometrics/svf/svf_streets_solar.gpkg`
(UMEP-validated by `scripts/validate_svf_against_umep.py`). Classified
each street observation point into:
`DEEP_CANYON` (SVF<0.15) · `SHELTERED` (0.15–0.30) · `INTERMEDIATE`
(0.30–0.50) · `OPEN` (>0.50).

| Site | n_pts | SVF p5 | p50 | p95 | DEEP | SHELTERED | OPEN |
|---|---|---|---|---|---|---|---|
| Vidigal | 6 876 | 0.01 | 0.33 | 0.72 | 29.5 % | 16.9 % | 28.1 % |
| Rocinha | 38 690 | 0.00 | 0.17 | 0.74 | **47.6 %** | 15.8 % | 19.7 % |
| Alemão | 47 508 | 0.03 | 0.37 | 0.78 | 20.6 % | 20.5 % | 32.6 % |
| Rio das Pedras | 16 905 | 0.02 | 0.18 | 0.65 | **44.1 %** | 27.8 % | 11.4 % |
| Maré | 84 147 | 0.09 | 0.54 | 0.99 | 10.8 % | 17.1 % | **53.1 %** |

**Why street SVF > grid SVF for ventilation interpretation.** Grid SVF
averages point samples to 10 m cells regardless of whether the cell
contains pedestrian space; rooftop samples (SVF→1) and degenerate
interior cells (SVF→0) contaminate the cell mean. Street SVF samples
*only* at observer points on the pedestrian network — the locus where
ventilation matters. The comparator stats are written into
`svf_streets_stats.json` per site.

Literature anchor: Yang et al. 2022 (SVF↔ground-level wind velocity
ratio). The CFD-ACH campaign supersedes this geometric proxy when it
lands.

**Outputs:**
- `outputs/<site>/morphometrics/svf/ventilation_openness_streets.gpkg`
  (5 files; SVF + categorical openness_class per street observation)
- `outputs/brisa_ventilation_fix/svf_streets_stats.json`

---

## 4. Recomputed four-state taxonomy shares

`build_diagnostic_map.py` classifies cells against the 2×2 of
{winter-sun < 2 h} × {ventilation fail}. Below: shares (%) of built
cells in each state under **four** corrected ventilation-axis choices.
The broken legacy axis is no longer reproducible (in-place overwrite);
its distribution is preserved in `lambda_f_repair_stats.json` where
~99 % of built cells fell into ventilation constraint + compound constraint.

### Vidigal
| Axis | adequate | sun-only | vent-only | compound |
|---|---|---|---|---|
| clipped λf > 0.40 (Macdonald) | 9.8 | 3.9 | 39.7 | 46.6 |
| clipped λf > within-site p75 (=2.10) | **41.9** | 33.1 | 7.6 | 17.4 |
| street SVF < 0.30 (sheltered) | **41.4** | 17.7 | 8.1 | 32.8 |
| H/W > 0.65 (skimming) | 22.3 | 5.2 | 27.2 | 45.3 |

### Rocinha
| Axis | adequate | sun-only | vent-only | compound |
|---|---|---|---|---|
| clipped λf > 0.40 | 7.6 | 2.2 | 21.6 | 68.6 |
| clipped λf > p75 (=3.27) | 27.4 | 47.6 | 1.9 | 23.1 |
| street SVF < 0.30 | 23.7 | 11.3 | 5.5 | **59.5** |
| H/W > 0.65 | 13.4 | 3.0 | 15.9 | **67.8** |

### Complexo do Alemão
| Axis | adequate | sun-only | vent-only | compound |
|---|---|---|---|---|
| clipped λf > 0.40 | 15.6 | 2.4 | 46.0 | 36.0 |
| clipped λf > p75 (=1.89) | **52.4** | 22.6 | 9.2 | 15.8 |
| street SVF < 0.30 | **52.8** | 11.0 | 8.8 | 27.3 |
| H/W > 0.65 | 33.1 | 4.8 | 28.6 | 33.6 |

### Rio das Pedras
| Axis | adequate | sun-only | vent-only | compound |
|---|---|---|---|---|
| clipped λf > 0.40 | 5.3 | 0.8 | 37.5 | 56.4 |
| clipped λf > p75 (=3.65) | 36.6 | 38.4 | 6.2 | 18.8 |
| street SVF < 0.30 | 31.8 | 2.6 | 10.9 | **54.6** |
| H/W > 0.65 | 20.7 | 2.1 | 22.1 | **55.1** |

### Maré
| Axis | adequate | sun-only | vent-only | compound |
|---|---|---|---|---|
| clipped λf > 0.40 | 13.0 | 0.4 | 58.2 | 28.4 |
| clipped λf > p75 (=2.73) | **60.7** | 14.3 | 10.5 | 14.5 |
| street SVF < 0.30 | **64.7** | 6.5 | 6.5 | 22.3 |
| H/W > 0.65 | 51.3 | 1.6 | 20.0 | 27.2 |

**What this tells the paper.**

1. The **clipped λf with the Macdonald threshold is still saturating**
   for the favela density regime — the manuscript must not claim
   Macdonald 0.40 as a within-favela discriminator.
2. The **H/W and street-SVF axes give coherent, discriminating
   results**: across sites the **compound constrainture share** under H/W
   correlates very well with under street SVF (Vidigal 45 / 33,
   Rocinha 68 / 60, Alemão 34 / 27, Rio das Pedras 55 / 55,
   Maré 27 / 22). Two independent geometric channels triangulate the
   same urban-form story.
3. The **within-site p75 λf axis** is a defensible *relative* threshold
   for spatial mapping — keeps the absolute-threshold honesty in §2
   while still letting Fig 0.4-style maps identify the worst quartile
   per site.

Outputs: `outputs/brisa_ventilation_fix/taxonomy_shares.json`.

---

## 5. Status by prong

| Prong | Status | Recommendation to paper |
|---|---|---|
| A — repair grid λf | ✅ done | Use canonical `grid_metrics.{csv,gpkg}` (now corrected); cite Macdonald 1998; **do not** claim Macdonald 0.40 as a within-favela skimming pre-screen. Replace with within-site p75 for spatial maps. |
| B — H/W canyon proxy | ✅ done | This is now the **primary geometric ventilation pre-screen** for BRISA+. Cite Oke 1988 H/W > 0.65. Per-site shares above. |
| C — street SVF openness | ✅ done | Use as the **secondary anchor**; documents why street SVF supersedes grid SVF for ventilation. Cite Yang 2022. |

All three are validated; none blocked. The CFD-ACH campaign remains the
calibrated supersession path — these three are the corrected interim
proxies.

---

## 6. Files index

```
outputs/brisa_ventilation_fix/
  REPORT.md                              ← this file (mirrored to docs/)
  lambda_f_repair_stats.json
  hw_streets_stats.json
  svf_streets_stats.json
  taxonomy_shares.json

outputs/<site>/morphometrics/
  grid/grid_metrics.{csv,gpkg}           ← Prong A (in-place overwrite)
  canyon/hw_streets.gpkg                 ← Prong B
  svf/ventilation_openness_streets.gpkg  ← Prong C

src/urban_morphology.py                  ← clip_to_zone=True (default)
src/morphometry/indicators.py            ← clip_to_zone pass-through
tests/test_urban_morphology.py           ← regression test added

scripts/brisa_ventilation/
  01_repair_grid_lambda_f.py             ← Prong A
  02_hw_canyon_proxy.py                  ← Prong B
  03_svf_streets_openness.py             ← Prong C
  04_recompute_taxonomy.py               ← §4 taxonomy shares
```

Run order: `01 → 02 → 03 → 04`. Each script is idempotent.
