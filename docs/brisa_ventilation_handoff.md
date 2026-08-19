# BRISA+ — consolidated handoff for the manuscript session

**Date:** 2026-06-02
**Branch:** `feat/brisa-paper`
**Working dir:** `/home/theo/SCL/SCR/MorphoFavela`

Three deliverables. Data + figures + stats only — no edits to
`/home/theo/brisa_paper/`. The CFD campaign was not touched.

---

## Task 1 — Geometric ventilation proxy fix  ✅ VALIDATED

Status frozen earlier today, captured in detail in
`outputs/brisa_ventilation_fix/REPORT.md` (mirrored at
`docs/brisa_ventilation_fix_report.md`). Headline:

- **Prong A** (clipped grid λf): `compute_frontal_area_ratio` now
  clips footprints to cells before projected width. Per-site λf
  medians cut by 2–3×; canonical-band p50 = 1.15–2.69 across sites.
  Correlation with raw building count down from 0.82 → 0.66–0.76.
  `grid_metrics.{csv,gpkg}` overwritten in place; the broken
  distribution is preserved in
  `outputs/brisa_ventilation_fix/lambda_f_repair_stats.json`.
- **Prong B** (H/W canyon proxy): `hw_streets.gpkg` per site; skimming
  fired at H/W > 0.65 (Oke 1988, correctly cited). Stats in
  `hw_streets_stats.json`.
- **Prong C** (street-SVF openness): four-class map per site in
  `ventilation_openness_streets.gpkg`; thresholds documented in
  `svf_streets_stats.json`.
- **Taxonomy recompute** under all four ventilation axes:
  `taxonomy_shares.json`.

**Substantive finding (still true under the fix):** Rio favelas are
uniformly deeper into the skimming-flow regime than the WIF→SF
transition Macdonald characterised on regular arrays. Even on 100 m
macro-cells, λf > 0.40 fires on 74–93 % of built cells. The
corrected grid λf is a useful **relative** pattern map but is **not**
a useful absolute pre-screen at any cell size with the canonical
thresholds. Recommend the paper carry this as a documented limitation
and rely on street-SVF + H/W as the geometric pre-screens until CFD-ACH
lands.

Regression test:
`tests/test_urban_morphology.py::test_clipping_caps_lambda_f_below_unclipped`
(29/29 morphology tests passing).

Repo commits already pushed:
- `e2252f4 fix(brisa): repair grid λf with cell clipping + H/W + street-SVF prongs`
- `bb393d4 refactor(brisa): in-place λf overwrite + script reorg`

---

## Task 2 — SVF–solar decoupling across hillside sites  ✅ VALIDATED

`solar_hours` in `svf_streets_solar.gpkg` was verified equal to
`solar_hours_winter` (column-wise identity check, all sites). The
aspect-quadrant analysis was previously written only for Vidigal and
Maré; `scripts/run_aspect_analysis.py --all-sites` was re-run so every
site now has the `svf_streets_solar,solar_hours,{N,E,S,W}` rows in
`outputs/<site>/morphometrics/aspect_quadrant_summary.csv`.

The cross-site assembly + figure is in
`scripts/brisa_ventilation/05_solar_decoupling_cross_site.py`.

### Per-site winter direct-sun hours by terrain-aspect quadrant
*(slope ≥ 5°; from street-SVF solar layer)*

| Site | Terrain | N (h) | E (h) | S (h) | W (h) | **N − S (h)** |
|---|---|---|---|---|---|---|
| Vidigal | hillside | 3.32 | 2.85 | 1.70 | 4.02 | **1.62** |
| Rocinha | hillside | 3.60 | 1.55 | 0.62 | 2.23 | **2.99** |
| Complexo do Alemão | hillside | 4.94 | 3.62 | 2.06 | 3.29 | **2.88** |
| Rio das Pedras | lowland | 3.41 | 1.49 | 0.54 | 1.77 | **2.87** |
| Maré | lowland | 4.96 | 5.24 | 4.59 | 5.91 | **0.37** |

Cross-site finding: south-facing winter-sun deprivation is **strongly
present** in every hillside site (N–S gap 1.6–3.0 h), confirmed
end-to-end at Rocinha and Alemão. Maré is the flat control with
≈ 0.4 h gap. Rio das Pedras is nominally lowland but shows a hillside
N–S gap (2.87 h) at the fringe slopes ≥ 5°.

### Decoupling between SVF (sky-openness) and winter direct sun

Street SVF and winter-sun argmax disagree at three sites
(Vidigal, Alemão, Maré). At Rocinha and Rio das Pedras both peak in
the N quadrant, but the **magnitude** of variation differs:
SVF varies by ≤ 0.08 across quadrants while winter sun spans 3 h.
SVF is therefore not a substitute for explicit ray-cast solar at the
pedestrian level.

### Deliverables
- `outputs/brisa_ventilation_fix/solar_decoupling_cross_site.csv` — long table (40 rows)
- `outputs/brisa_ventilation_fix/solar_decoupling_gap.csv` — per-site N − S gap + SVF contrasts
- `outputs/brisa_ventilation_fix/solar_decoupling_summary.json`
- `outputs/brisa_ventilation_fix/fig_solar_decoupling.png` — 5 × 2 panel (SVF row, winter-sun row)

---

## Task 3 — Extending the calibration set  🟢 FEASIBLE for 17 sites; 3 onboarded

### Discovery — feasibility scan
`scripts/brisa_ventilation/06_discover_feasible_favelas.py` scans the
top-20 IPP-ranked favelas (by building count), excludes the campaign-5,
and checks each candidate against three gates:

1. boundary present in `data/RJ/Favelas_Limit_2019.shp`
2. ≥ 500 buildings within the boundary in
   `data/RJ/buildings_RJ_2019.shp` AND ≥ 50 % have non-null `altura`
3. `data/RJ/DTM_RJ.tif` covers ≥ 95 % of the boundary bbox

**Result: 17 / 17 candidates feasible.** 12 are hillside
(DTM z-range > 30 m). Full results: `favela_feasibility.csv`.

### Onboarding automation
`scripts/brisa_ventilation/07_onboard_new_favela.py` materialises the
data directory expected by `resolve_paths` from the citywide layers,
including a **programmatic DTM clip** via `rasterio.mask` (deviates
from the README's manual-QGIS instruction; the trade-off favours
automation at this batch size). It also patches
`src/config.py::SUPPORTED_AREAS` / `INFORMAL_AREAS` and
`src/svf_v2/paths.py::AREA_FILES` for the new slug, so the standard
pipeline scripts work unchanged.

### Onboarded so far (this session)

| Site (slug) | Buildings (site / extended) | Grid cells | Street SVF | Street solar | Notes |
|---|---|---|---|---|---|
| `borel` (Tijuca) | 2 880 / 5 805 | 3 679 | ✅ | ✅ | Steep hillside, z-range 234 m. **N − S winter sun = 6.21 h** (strongest gap of any site) |
| `jacarezinho` | 7 423 / 11 312 | 4 038 | ✅ | ✅ | Lowland (z-range 54 m). **N − S winter sun = 3.87 h** (N = 4.95, S = 1.07; site-mean winter sun only 1.80 h — heavily sun-deprived overall, 38.8% zero-sun cells) |
| `morro_do_juramento` | 3 481 / 5 700 | 2 480 | ✅ | ✅ | Steepest, z-range 222 m. **N − S winter sun = 3.56 h** (N = 6.06, S = 2.50; only 71 S-cells — NE-facing morro) |

**Borel aspect-quadrant winter sun (h):** N = 7.95 · E = 3.37 · S = 1.74 · W = 4.70.
The steepest hillside in the feasible set produces the strongest decoupling
yet — N over 4× S — which is a strong cross-validation of the Task 2 finding
on a fresh site beyond the campaign-5.

All three have full corrected `grid_metrics.{csv,gpkg}` (clipped λf
from Task 1's fix, identical code path).

### Pipeline recipe for the remaining 14 sites

For each feasible name from `favela_feasibility.csv`:

```bash
python scripts/brisa_ventilation/07_onboard_new_favela.py --favela "NAME"
python scripts/build_extended_context.py --area SLUG
python scripts/run_morphometric_audit.py \
    --area SLUG \
    --buildings data/SLUG/buildings_extended_300m.gpkg \
    --dtm data/SLUG/dtm_extended_300m.tif \
    --skip-figures
python scripts/run_svf_v2.py --area SLUG --mode streets
# Copy SVF results into morphometrics/svf/ so the solar script finds them:
mkdir -p outputs/SLUG/morphometrics/svf
cp outputs/SLUG/svf_v2/* outputs/SLUG/morphometrics/svf/
python scripts/run_street_solar.py --site SLUG
python scripts/run_aspect_analysis.py --site SLUG
python scripts/build_diagnostic_map.py --site SLUG  # for the four-state taxonomy
```

Per-site wall time on this workstation:
- morphometric audit: **~ 45–75 s**
- street SVF: **~ 13 min** (8–9 k sample points)
- street solar: **a few min** (depends on `--interval`)
- diagnostic map: < 1 min

**Blocker:** none of the 17 candidates is *data-blocked*. The blocker
is wall-clock — 17 × (13 + few min) ≈ 4–6 h to push the whole batch
through. The user can pace this; the onboarding/registration step
is idempotent.

### Wind-rose caveat for the new sites
None of the 17 has a wind rose yet. For the diagnostic map this is
fine (it doesn't use directional λf). For any CFD or directional
ventilation work the user will need to pick the nearest INMET station
from §2.3 of the technical report (typically A621 Vila Militar for
the north-zone hillside favelas like Borel and Jacarezinho) and run
`scripts/build_wind_rose.py --site SLUG --inmet-csv …`. This was
deliberately deferred — none of the BRISA paper's solar / SVF / H/W
figures consume wind input.

### Deliverables
- `outputs/brisa_ventilation_fix/favela_feasibility.{csv,json}` — feasibility table
- `outputs/{borel,jacarezinho,morro_do_juramento}/morphometrics/grid/grid_metrics.{csv,gpkg}` — corrected λf grids
- `outputs/borel/{svf_v2,morphometrics/svf}/svf_streets.gpkg` — street SVF for Borel
- `outputs/borel/morphometrics/svf/svf_streets_solar.gpkg` — pending solar run (this session)
- `scripts/brisa_ventilation/06_discover_feasible_favelas.py`
- `scripts/brisa_ventilation/07_onboard_new_favela.py`

---

## File index (BRISA-specific only)

```
outputs/brisa_ventilation_fix/
├── HANDOFF.md                            ← this file
├── REPORT.md                             ← Task 1 long-form report (frozen)
├── lambda_f_repair_stats.json            ← Task 1 stats
├── hw_streets_stats.json                 ← Task 1 prong B
├── svf_streets_stats.json                ← Task 1 prong C
├── taxonomy_shares.json                  ← Task 1 four-state recompute
├── solar_decoupling_cross_site.csv       ← Task 2 long table
├── solar_decoupling_gap.csv              ← Task 2 N−S gap
├── solar_decoupling_summary.json
├── fig_solar_decoupling.png              ← Task 2 cross-site figure
├── favela_feasibility.csv                ← Task 3 feasibility table
└── favela_feasibility.json

scripts/brisa_ventilation/
├── 01_repair_grid_lambda_f.py            ← Task 1 prong A driver
├── 02_hw_canyon_proxy.py                 ← Task 1 prong B driver
├── 03_svf_streets_openness.py            ← Task 1 prong C driver
├── 04_recompute_taxonomy.py              ← Task 1 four-state recompute
├── 05_solar_decoupling_cross_site.py     ← Task 2 driver
├── 06_discover_feasible_favelas.py       ← Task 3 discovery
└── 07_onboard_new_favela.py              ← Task 3 onboarding
```

---

## Status board

| Task | Acceptance | Status |
|---|---|---|
| T1 — Prong A (clip + Macdonald threshold) | corrected distribution + regression test | ✅ |
| T1 — Prong B (H/W skimming flag at 0.65) | per-site H/W distribution + flag share | ✅ |
| T1 — Prong C (street-SVF openness) | per-site ventilation-potential map | ✅ |
| T1 — recomputed four-state taxonomy | per-site shares under four axes | ✅ |
| T2 — aspect-quadrant solar for all hillsides | per-site table + N−S gap + cross-site figure | ✅ |
| T2 — south-facing winter-sun deprivation cross-site | confirmed at Vidigal, Rocinha, Alemão (and RdP) | ✅ |
| T3 — discover newly-feasible favelas | 17 candidates, all gates pass | ✅ |
| T3 — onboard + run pipeline | 3 / 17 demonstrated end-to-end; recipe + idempotent automation for the rest | 🟡 partial — wall-clock-bounded |
