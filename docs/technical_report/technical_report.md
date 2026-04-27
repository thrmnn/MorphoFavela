# Brisa+ Technical Report

## Urban Morphology and CFD Patch Sampling for Wind Simulation Across Five Informal Settlements in Rio de Janeiro

**Version 1.0 — 2026-04-13**
**Internal technical report (pre-CFD phase)**

---

## Executive Summary

This report documents the morphometric analysis and CFD sampling pipeline
developed for five informal settlements in Rio de Janeiro: Vidigal, Rocinha,
Rio das Pedras, Complexo do Alemão, and Maré. The work supports the Brisa+
project, which aims to quantify pedestrian-level wind conditions in informal
urban fabric and link them to health-relevant outcomes (thermal comfort,
pollutant dispersion, natural ventilation).

At time of writing, the pipeline has produced:

- **98,435 building footprints** compiled from site-specific and city-wide
  cadastral data, with defensive geometry repair and extended context beyond
  each favela boundary.
- **82,314 grid cells at 10 m resolution** covering the five sites, each
  characterised by 20+ morphometric indicators including Sky View Factor,
  plan area density, 8-direction frontal area density, volumetric porosity,
  height variability, slope, aspect, and street orientation entropy.
- **119 CFD simulation patches** selected via stratified sampling across
  twelve morphometric strata (SVF × slope × λp) with maximin geographic
  spacing and SVF-priority weighting to oversample health-relevant low-sky-
  view conditions.
- **A complete CFD integration pipeline** (`src/cfd_integration/`,
  46 passing tests) ready to ingest OpenFOAM results when the simulation
  campaign completes in the parallel CFD repository.

The pipeline is fully reproducible from the committed scripts; all inputs
are documented and all intermediate outputs preserved in the canonical
repository layout. This document summarises the methodology, validates
the sampling design, and specifies the interface between this repository
and the CFD execution environment.

---

## 1. Study Sites

Five favelas were selected to span the morphological typologies of Rio
informal settlements:

| Site | Area | Type | Buildings (extended) | 10 m cells | Mean building height |
|------|-----:|------|--------------------:|-----------:|--------------------:|
| Vidigal | 0.30 km² | hillside | 4,600 | 3,169 | 6.3 m |
| Rocinha | 0.80 km² | hillside | 14,443 | 8,972 | 5.7 m |
| Rio das Pedras | 0.70 km² | flatland | 11,276 | 7,046 | 5.9 m |
| Complexo do Alemão | 1.97 km² | mixed | 28,783 | 19,708 | 6.5 m |
| Maré | 4.34 km² | flatland | 39,333 | 43,419 | 6.4 m |
| **TOTAL** | **8.11 km²** | | **98,435** | **82,314** | — |

*Extended buildings = site footprints plus context buildings within a 300 m
buffer (see Section 3.2). Typology assignment drives wind-regime analysis:
hillside sites span 0–45° slopes; flatland sites cluster near 0°.*

![Figure 1. Study sites overview.](figures/fig01_study_sites.png)

**Figure 1.** Overview of the five campaign sites on the Rio de Janeiro
hillshade. The main panel shows the full metropolitan extent with all 1,074
registered favelas outlined in light grey; the five campaign sites are
highlighted with typology-coloured fills. Right panels show each site's
building footprints at a consistent visual style, with favela boundary and
building count. Vidigal and Rocinha occupy the steep south-zone hillslopes
near Sugarloaf; Maré and Complexo do Alemão are in the flat north-zone
bayside plain; Rio das Pedras is in the west-zone plain.

Cidade de Deus was initially considered but excluded from the CFD campaign
due to a documented building-data defect (see
`.claude/…/memory/project_cdd_data_bug.md`); its SVF values saturated at
≈ 0.90, indicating missing building footprints in the computational scene.

---

## 2. Data Sources

### 2.1 Site-specific data

Each site has a `data/{site}/raw/` directory containing:

- **Building footprints** (.shp, 3D polygons) with attributes `altura`
  (height above base), `base` (base elevation), `topo` (top elevation).
  CRS: EPSG:31983 (SIRGAS 2000 / UTM zone 23S).
- **Favela boundary** (.shp) from the 2019 Rio municipal cadastre.
- **Digital terrain model** (.tif) at 5 m resolution clipped to the site
  extent during data preparation (manually in GIS; not regenerated
  code-side per project memory).
- **Road network** (.shp) from the municipal `Logradouros` dataset,
  clipped and projected per site.

### 2.2 City-wide data (shared across sites)

Used to extend the building context and ensure continuous terrain for CFD
domains that exceed individual favela boundaries.

- **`data/RJ/buildings_RJ_2019.shp`**: 2,362,806 building footprints for
  the entire municipality. Same schema as per-site files (subset extracted
  from this source).
- **`data/RJ/DTM_RJ.tif`**: 72 km × 37 km DTM at 5 m resolution covering
  all of Rio.
- **`data/RJ/Favelas_Limit_2019.shp`**: 1,074 registered favelas, used
  for the Figure 1 overview.

### 2.3 Wind forcing (INMET)

Annualised wind roses are required to weight CFD results across the eight
cardinal directions. Each site has a `data/{site}/wind_rose.json` file
whose schema is defined by `src.cfd_integration.schema.WindRose` — frequencies
and mean speeds per direction, plus provenance metadata (station id,
coordinates, time window, observation count, calm fraction, anemometer
reference height, and a `quality_flag` ∈ {`measured`, `gap-filled`,
`placeholder-prior`}).

**Current status (April 2026).** All five `wind_rose.json` files now
carry measured hourly observations across the full 2015–2024 window
and are tagged `quality_flag: "measured"`. The four hillside / inland
sites use INMET BDMEP records; Maré uses the Iowa State ASOS METAR
archive of SBGL Galeão. Sample sizes range from n = 64,088 (A636
Jacarepaguá, which only began reporting in August 2017) to n = 89,439
(SBGL). Per-site directional frequencies, mean speeds, and calm
fractions are shown in **Figure S5**
(`docs/technical_report/figures/figS5_wind_roses.png`):

| Site (station) | n | Window | Calm | Prevailing dir | Notes |
|---|---|---|---|---|---|
| Vidigal / Rocinha (A652 Forte de Copacabana) | 85,103 | 2015–2024 | 1.4 % | E (30 %), W (23 %) | Coastal cliff: bimodal sea-breeze and reverse-channel along the Dois Irmãos saddle. |
| Rio das Pedras (A636 Jacarepaguá) | 64,088 | 2017–2024 | 46.3 % | SW (26 %), W (20 %) | Sheltered basin, weak winds (Ū ≈ 1.3 m/s); high calm fraction reflects the Jacarepaguá lowland regime, not measurement gaps. |
| Complexo do Alemão (A621 Vila Militar) | 86,019 | 2015–2024 | 33.1 % | N (26 %), SW (19 %) | North-zone interior; bay sea-breeze + post-frontal SW pattern. |
| Maré (SBGL Galeão METAR) | 89,439 | 2015–2024 | 3.7 % | SE (26 %), E (19 %) | Bay regime; continuous METAR coverage gives the largest sample. |

**Recommended stations** (verified April 2026 against the INMET
catalogue, daily-graph URLs, and the published TMY paper for A652):

| Site | Station | Code | Coords (lat, lon) | Class | Notes |
|------|---------|------|-------------------|-------|-------|
| Vidigal | Forte de Copacabana | A652 | −22.988, −43.190 | coastal | Nearest unobstructed coastal reference (~5 km E). CFD captures the Dois Irmãos lee-side locally; inflow rose stays at A652. |
| Rocinha | Forte de Copacabana | A652 | −22.988, −43.190 | coastal | Provides the unobstructed SE→NE driver. Valley channelling is resolved by the CFD itself. |
| Rio das Pedras | Jacarepaguá | A636 | −22.99, −43.37 | plain | Colocated with the Jacarepaguá lowland. |
| Complexo do Alemão | Vila Militar | A621 | −22.86, −43.41 | urban interior | Closest north-zone station (~8 km W). Corrects the earlier placeholder recommendation of A602 Marambaia, which is geographically mismatched (southwest coast, not north zone). |
| Maré | SBGL Galeão METAR (preferred); A652 (INMET fallback) | — / A652 | −22.81, −43.25 / −22.988, −43.190 | bayside | Galeão airport METAR via the Iowa State ASOS archive is the best match for the bay regime; METAR ingestion is not yet implemented in `build_wind_rose.py`. A652 is the INMET fallback. |

**The earlier placeholder attributed A652 to "Alto da Boa Vista" — that
name belongs to the municipal Alerta Rio network, not INMET. A652 is
Forte de Copacabana. The station table is corrected here.**

**Data acquisition.** INMET publishes one nationwide yearly ZIP
(~100 MB) at
`https://portal.inmet.gov.br/uploads/dadoshistoricos/{YEAR}.zip`, live
and unauthenticated (browser User-Agent required). Each ZIP contains
one CSV per automatic station for the year. The ingestion pipeline is
documented in `src/cfd_integration/README.md` and implemented in
`scripts/build_wind_rose.py from_inmet_csv`.

BDMEP CSV format: `sep=;`, `decimal=,`, latin-1 encoding, 8-row
metadata header, missing values encoded as −9999, anemometer at z =
10 m. Calm periods (|U| < 0.5 m/s or direction = NaN) are excluded
from direction binning but recorded in `calm_fraction` so they are not
hidden.

**Neutral-stability assumption.** The k-ω SST inlet uses a log-law
profile that implicitly assumes a neutral atmospheric boundary layer.
For Rio, daytime unstable convection and evening stable inversions
bias the stagnation metric. This limitation is accepted for the
screening campaign and documented in §10. A future campaign could
stratify the rose by stability class (e.g., day vs night, or
Pasquill-Gifford class).

### 2.4 Coordinate reference system

All vector and raster data are in **EPSG:31983 (SIRGAS 2000 / UTM zone 23S)**,
verified on load. Mixed-CRS inputs are reprojected automatically in the
pipeline.

---

## 3. Data Preparation

### 3.1 Building footprint processing

Raw footprints undergo three processing steps before use:

1. **Topology validation.** Shapely's `is_valid` flag is checked on all
   geometries. Rocinha was found to have 10 self-intersecting polygons
   (0.07 %) out of 13,807; all other sites had zero invalid geometries.
   All invalids are self-intersections (common artefact of municipal GIS
   digitisation) and are repaired by `buffer(0)`, which corrects the
   geometry with sub-square-metre area change. The repair is applied
   defensively in both the extended-context builder and the morphometric
   pipeline before any spatial operation. Full audit:
   `outputs/comparative/audits/rocinha_topology_audit.md`.

2. **Height validation.** `altura` values are checked for negatives
   (none) and unrealistic extremes (one building > 50 m — a legitimate
   high-rise in context data near a favela).

3. **CRS enforcement.** If source CRS differs from EPSG:31983, the
   dataset is reprojected.

### 3.2 Extended building context

**Problem.** Each CFD simulation domain is a circular region of 250 m
radius around the analysis patch. When the patch is near the favela
perimeter, the domain extends beyond available building data, leaving the
outer annulus without context. A naïve sampling restricted to patches
fully within the favela excludes 55 % of candidate cells at Vidigal.

**Solution.** The city-wide RJ building dataset is clipped to a 300 m
buffer around each favela boundary and merged with the site-specific
dataset, preferring site data where footprints overlap (deduplication via
spatial join on the boundary polygon).

![Figure S2. Context extension validation.](figures/figS2_context_extension.png)

**Figure S2.** Context extension for Vidigal. Left: site buildings (blue)
within the boundary and context buildings (orange) from city-wide data
within the 300 m buffer. Middle: candidate-pool sensitivity to buffer
distance — 0 m yields 980 eligible cells (31 %); 300 m yields 1,697
(54 %); 400 m only adds 77 more (diminishing returns). Right: pilot patch
locations on the extended footprint — without the extension, patches
cluster on the interior 600 m × 250 m; with it, they span the full
1,040 m × 310 m.

Extended buildings per site:

| Site | Site buildings | Context buildings | Extended total |
|------|--------------:|-----------------:|---------------:|
| Vidigal | 3,695 | 905 | 4,600 |
| Rocinha | 13,807 | 636 | 14,443 |
| Rio das Pedras | 10,729 | 547 | 11,276 |
| Complexo do Alemão | 21,729 | 7,054 | 28,783 |
| Maré | 37,199 | 2,134 | 39,333 |

Implementation: `scripts/build_extended_context.py`. The 300 m buffer was
selected as the inflection point of a sensitivity scan (200 m / 300 m /
400 m); 300 m reduces domain-extent exclusion to ≤ 20 % at all sites
while adding modest file size.

### 3.3 Extended terrain

The site-specific DTM is merged with the city-wide RJ DTM (both 5 m
resolution) over the same 300 m buffer. The site DTM takes priority in
overlap regions (`rasterio.merge.merge` with `method="first"`). A
post-merge sanitisation step replaces any residual float32-max sentinel
values (historical nodata artefact) with a clean nodata = −9999 and
declares it in the output profile. This was found to be necessary after
the first run produced downstream warnings about values of order 3.4×10³⁸
in the merged raster.

---

## 4. Morphometric Grid

### 4.1 Grid construction

A regular 10 m grid is generated covering each extended-buildings extent,
then clipped to the favela boundary so that only cells whose centroid
falls within the settlement are retained. Cells range from 3,169 at
Vidigal to 43,419 at Maré, with total 82,314 cells across the five sites.

### 4.2 Indicators

Per 10 m cell, the pipeline computes:

**Sky View Factor (SVF).** The fraction of hemispheric sky visible from
pedestrian height. Computed at 1.5 m above the ground surface via
Tregenza 145-patch ray-casting on a sample grid of passageway points;
the per-cell value is the mean of all sample points whose centroid falls
within the cell. An `svf_count` column records the number of contributing
sample points per cell (mean ≈ 13, minimum 1). Cells with no contributing
samples receive NaN.

Critically, SVF is **aggregated from passageway samples, not computed at
grid-cell centroids**. Centroid computation would bias SVF toward 0 for
cells whose centroid falls inside a building; passageway aggregation
avoids this artefact. A diagnostic verification confirmed no spurious
SVF = 0 spikes (see `src/morphometry/grid.py::_aggregate_svf_to_grid`
and `docs/technical_report/validation.md`).

**Plan area density (λp, BCR).** Building footprint area ÷ cell area.
Capped at 1.0 to prevent over-counting from overlapping footprints in
dense informal fabric.

**Frontal area density (λf).** Projected vertical surface area per unit
horizontal cell area, computed for eight wind directions (N, NE, E, SE,
S, SW, W, NW); stored as `lambda_f_{direction}` plus `lambda_f_mean` and
`lambda_f_max`. Directional λf is the primary morphological input to
wind-canopy drag models.

**Volumetric porosity.** `1 − (Σ building_volume_in_cell) / (cell_area ×
H_mean)`. Characterises the void fraction in the urban canopy; strongly
anti-correlated with λp (r ≈ −0.95 across sites, see Section 5.3).

**Height variability (σH).** Standard deviation of building heights in
the cell. Cells with < 2 buildings receive NaN.

**Mean height (H_mean).** Arithmetic mean of contributing building
heights.

**Street orientation entropy.** Shannon entropy of street-segment
bearings clipped to the cell (bearings folded to [0, 180°) because
streets are bidirectional), normalised to [0, 1]. Measures how ordered
the street grid is: 0 = single dominant direction, 1 = uniform
distribution across 36 bins.

**Terrain slope and aspect.** Derived from the merged DTM via numpy
gradient, sampled within each cell and averaged. Aspect uses a circular
mean to avoid discontinuity at 0°/360°.

Implementation: `src/morphometry/grid.py`. The computation is
deterministic and takes ~30–80 s per site (longest is Complexo do Alemão
at ~85 s) on standard hardware.

### 4.3 Distributions across sites

![Figure 2. Morphometric distributions.](figures/fig02_morphometric_distributions.png)

**Figure 2.** Distributions of the five primary indicators (SVF, λp, λf,
σH, slope) across all five sites. Violins show the full distribution per
site; white dots are medians and dark bars span the IQR. Dashed
horizontal lines mark stratum boundaries (SVF at 0.15 and 0.30;
λp at 0.5; slope at 15°). Background shading groups hillside (left,
warm) and flatland (right, cool) sites.

Key patterns:

- **Slope clearly separates typologies.** Hillside sites (Vidigal,
  Rocinha) have most cells at 15–35°; flatland sites (Rio das Pedras,
  Maré) concentrate below 5°; Complexo do Alemão spans both regimes.
- **λp distributions are similar across sites** (median 0.4–0.7),
  indicating comparable building-packing density regardless of terrain.
- **SVF distributions differ substantially.** Rocinha's dense canopy
  produces a strongly left-skewed distribution (median 0.27); Maré's
  more open layout has a right-skewed distribution (median 0.53).
- **σH is small everywhere** (median 2–3 m), reflecting the typical 2–3
  storey construction.

### 4.4 Spatial visualisation

![Figure 5. Morphometric maps.](figures/fig05_morphometric_maps.png)

**Figure 5.** Spatial maps of SVF, λp, and slope for Vidigal (hillside)
and Maré (flatland), with hillshade basemap and building outlines. Colour
bars show indicator ranges; white contour lines on colourbars mark the
stratification thresholds.

Notable observations:

- Vidigal's SVF map shows clear low-SVF corridors along the hillslope
  ridgelines where canopy density is highest (dark cyan in upper-left
  panel).
- Maré's λp is near-uniform at 0.7–1.0 across most of the settlement,
  reflecting the tight block structure of the original housing project
  layout.
- Slope maps confirm Vidigal's topography (green-yellow, 20–40°) vs
  Maré's flatness (purple, 0–3° except at engineered berms near the
  bay).

### 4.5 Indicator correlations

![Figure S1. Correlation matrices.](figures/figS1_correlation_matrices.png)

**Figure S1.** Pearson correlation matrices for the eight primary
indicators at each site. Values are annotated in each cell; pairs with
|r| > 0.7 are framed in black. Colour scale is fixed across all panels
so patterns are directly comparable.

- **λp ⇄ porosity** is the strongest correlation everywhere (r ≈ −0.95),
  which motivates dropping porosity from the manuscript's distribution
  figure (Figure 2) to avoid redundancy.
- **SVF ⇄ λp** correlation is strong but not perfect (r ≈ −0.70 to
  −0.85): variation in building *height* and arrangement breaks the
  λp↔SVF link at fine scales. This is the structural-coupling finding
  developed in Section 5.
- **σH correlates with H_mean** (r ≈ 0.5) because denser cells with more
  buildings have more variance.
- **Slope is nearly uncorrelated with morphology**, confirming that
  terrain and urban form are independent stratification axes — a
  necessary condition for the 12-stratum sampling scheme.

### 4.6 Resolution sensitivity

![Figure S3. Resolution sensitivity.](figures/figS3_resolution_sensitivity.png)

**Figure S3.** 10 m (solid coloured curves) vs 20 m (dashed black
curves) morphometric distributions across all five sites and five
indicators. The 10 m grid captures multimodal structure in λp and λf
that 20 m aggregation smooths away — particularly visible in Complexo do
Alemão and Rio das Pedras where the 20 m curves flatten features that
clearly exist in the 10 m data.

This analysis justifies the 10 m resolution as the production grid:

- The key finding is preserved — distribution shape is similar —
  confirming that 10 m is not noise.
- But fine-scale multimodality relevant to CFD boundary conditions is
  only resolved at 10 m.
- Computational cost at 10 m is manageable (< 85 s per site).

Supplementary variants (Vidigal-only upscaled-vs-native scatter, and
side-by-side difference maps) are archived in
`outputs/paper_figures/exports/_variants/` and regenerable with
`figS3_resolution_sensitivity.py --variants`.

---

## 5. Cross-Site Morphology

### 5.1 SVF–λp structural coupling

![Figure 3. SVF–λp coupling.](figures/fig03_svf_lambda_coupling.png)

**Figure 3.** Central panel: 2D kernel density estimation of cells in
(SVF, λp) space, grouped by typology. Solid contours enclose 50 % of
each typology's cells; dashed contours enclose 90 %. Terracotta =
hillside (Vidigal, Rocinha); olive-gold = mixed (Complexo do Alemão);
steel blue = flatland (Rio das Pedras, Maré). Marginals show per-site
SVF and λp distributions. The salmon-shaded region in the lower-left
(SVF < 0.15, λp < 0.50) is **structurally empty** — only 102 cells in
35,144 (0.3 %) — because low sky view requires high plan density in
realistic urban fabric. The strong negative correlation (r = −0.80
across all pooled cells) reflects this coupling. The three typologies
occupy distinct but partially overlapping regions of the feature space:
hillside cells cluster at higher λp for a given SVF, consistent with
taller canopies; flatland cells spread to lower λp.

This figure is the central conceptual claim for the manuscript: the
morphology of informal settlements occupies a constrained subset of the
(SVF, λp) plane, and stratified CFD sampling must respect this
constraint (empty strata are not allocation failures but physical
impossibilities).

### 5.2 Typology summary statistics

| Metric | Hillside (VDG+ROC) | Mixed (CDA) | Flatland (RdP+MAR) |
|--------|-------------------:|------------:|-------------------:|
| Slope, median | 22° | 8° | 0.8° |
| SVF, median | 0.31 | 0.37 | 0.50 |
| λp, median | 0.62 | 0.50 | 0.56 |
| λf_mean, median | 0.38 | 0.29 | 0.19 |
| σH, median | 2.4 m | 1.7 m | 2.3 m |

Hillside sites have the highest median frontal area density because
buildings stacked on slopes present more vertical surface area per
horizontal unit than their flatland counterparts. This has direct
implications for wind-canopy drag and for the prevalence of leeward
recirculation zones that CFD will quantify.

---

## 6. CFD Patch Sampling

### 6.1 Sampling design

Each CFD simulation corresponds to one **analysis patch** of interest
— a 100 m-diameter circle (radius 50 m, area ≈ 7 854 m²) — embedded in
a 250 m-radius circular domain that provides flow-development context
(per Blocken 2015 and COST Action 732). The circular patch shape
matches the cylindrical symmetry of the CFD domain, avoids corner
artefacts in morphometric averaging, and produces isotropic coverage
independent of building-grid orientation. To characterise the
morphological diversity of the five sites with a finite simulation
budget, patches are allocated across **twelve stratification bins**:

- SVF: 3 bins (SVF < 0.15, 0.15 ≤ SVF < 0.30, SVF ≥ 0.30)
- Slope: 2 bins (< 15°, ≥ 15°)
- λp: 2 bins (< 0.5, ≥ 0.5)

giving 3 × 2 × 2 = 12 strata, named e.g. `SVF1_SLP2_LP2` for
SVF < 0.15 + slope ≥ 15° + λp ≥ 0.5.

### 6.2 Eligibility filter

A cell is an eligible patch centre if:

1. The 250 m CFD domain is fully covered by available building data
   (the extended buildings from Section 3.2). Measured as the fraction
   of the circular domain that intersects the convex hull of the
   extended footprints; threshold 0.7.
2. Within the 100 m-diameter circular analysis patch, building
   footprint coverage (∑ intersection area ÷ π·50² ≈ 7 854 m²) ≥ 0.5.
   This also implicitly rejects patches clipped by the site boundary,
   where the missing area fails the coverage threshold.
3. All three stratification indicators are non-null.

Filter outcomes per site:

![Figure: candidate pool.](figures/fig_candidate_pool.png)

**Figure (candidate pool).** Stacked bars showing eligible cells (green)
vs excluded cells per filter. Two exclusion sources: domain-coverage
failure (red, dominated by edge cells in Vidigal and Rio das Pedras)
and low-building-coverage failure (orange, dominant in Maré because of
large residential block interiors without continuous footprint).

With the 300 m building-context extension, the domain-coverage
exclusion drops from 55 % (no extension) to 0–20 % depending on
settlement shape. The remaining exclusions are legitimate morphology:
Maré's 20 % overall eligibility rate reflects its lower footprint
coverage per 100 m patch, not a sampling artefact.

### 6.3 Selection algorithm

For each site, patches are chosen in two phases:

**Phase 1 — Pilot (12–15 patches per site).** Within each non-empty
stratum, at least one patch is allocated; an additional bonus is given
to the three most populated strata. Patches are chosen by **greedy
maximin geographic distance** starting from the two most distant
eligible cells; subsequent picks maximise minimum distance to already-
selected patches (and to patches of other strata selected earlier, so
that the cross-stratum spacing constraint is respected). Minimum
spacing: 80 m between patch centres. With 100 m-diameter patches, two
patches at the minimum centre-to-centre distance overlap in a lens of
≈ 817 m² (≈ 10.4 % of each patch area); this residual overlap is
accepted because the patches are evaluated independently in separate
CFD simulations and no joint statistics are computed over overlapping
footprints.

**Phase 2 — Campaign (incremental to 22–25 patches per site).** Pilot
patches are held fixed and treated as immovable seeds; additional
patches are chosen with **SVF-priority weighting** to oversample the
health-relevant low-SVF strata:

- SVF1 (< 0.15) → ×2.0 multiplier
- SVF2 (0.15–0.30) → ×1.0
- SVF3 (≥ 0.30) → ×0.8

The effective allocation is proportional to the weighted eligible count
per stratum, subject to per-stratum minimums (≥ 1 patch if any cells
exist) and the 80 m spacing constraint against the union of pilot and
newly-selected patches.

Implementations: `scripts/run_pilot_sampling.py` (Phase 1),
`scripts/run_campaign_sampling.py` (Phase 2),
`scripts/build_extended_context.py` (upstream).

### 6.4 Allocation results

![Figure 4. Sampling design.](figures/fig04_sampling_design.png)

**Figure 4.** (a–b) Feature-space coverage of the 119 selected patches
over the full grid-cell cloud for (SVF, slope) and (SVF, λp) axes.
Dotted lines mark stratum boundaries. (c) Horizontal bar chart of
patches per stratum, with segments coloured by site; numeric labels
give the stratum total. (d) Per-site maps showing analysis-patch
locations as coloured 100 m-diameter circles over grey building
footprints. All five sites shown at consistent visual scale with 200 m
scale bars.

![Figure: stratum heatmap.](figures/fig_strata_heatmap.png)

**Figure (strata heatmap).** Matrix of eligible cell counts (log₁₀
scale) and allocated patches per stratum per site. Cells with no
eligible candidates are marked with an em-dash. This makes the
partial-empty structure explicit: three strata are empty at one or more
hillside sites (SVF1_SLP1_LP1, SVF1_SLP2_LP1, SVF2_SLP1_LP1) because
Vidigal's narrow hillside settlement has no flat cells with sparse
footprints.

![Figure: campaign allocation summary.](figures/fig_campaign_allocation_summary.png)

**Figure (campaign allocation).** Total patches per stratum in the
final 119-patch campaign, segments coloured by site. SVF1_SLP1_LP2
(low sky view, flat, dense) has the highest patch count (21) after
SVF-priority weighting — this is the stratum most relevant to health
outcomes at the flatland sites and was intentionally oversampled.

![Figure: cross-site feature space.](figures/fig_campaign_cross_site_featurespace.png)

**Figure (cross-site feature space).** All 119 patches overlaid on all
five sites' grid-cell clouds across four feature-space projections
(SVF vs slope, SVF vs λp, slope vs λp, SVF vs σH). Site colours match
throughout. Verifies that sampled patches span the full morphological
envelope of the five sites without clustering in any projection.

![Figure: selected patches (all sites).](figures/fig_all_sites_patches.png)

**Figure (all sites overview).** 5-panel map of final patch locations
per site, each with hillshade basemap and patch outlines in site colour.

![Figure: patch metrics comparison.](figures/fig_patch_metrics_comparison.png)

**Figure (patch metrics).** Violin plots of morphometric values at the
119 selected patches, grouped by site. Confirms that the selection does
not over-represent or under-represent any single site's typical
conditions: each site's patch distribution spans its full grid-cell
distribution for each indicator.

Final per-site allocation:

| Site | Pilot | Campaign add | Total | Shortfall |
|------|------:|-------------:|------:|----------:|
| Vidigal | 12 | 10 | 22 | 0 |
| Rocinha | 15 | 10 | 25 | 0 |
| Rio das Pedras | 13 | 9 | 22 | −1 |
| Complexo do Alemão | 15 | 10 | 25 | 0 |
| Maré | 14 | 11 | 25 | 0 |
| **TOTAL** | **69** | **50** | **119** | **−1** |

Rio das Pedras has a one-patch shortfall against its 23-patch target:
the SVF2_SLP2_LP2 stratum has only 7 eligible cells, all within 80 m
of existing pilot patches, so no additional candidate could be placed
without violating the minimum-spacing constraint. This is a genuine
morphological limitation (the stratum is physically scarce) and was
accepted rather than relaxing the spacing.

### 6.5 Blocken compliance

For each selected patch, the maximum building height within the 100 m
analysis zone is recorded (`H_max_analysis`) and the Blocken (2015)
minimum-fetch requirement is computed as `5 × H_max`. All 119 patches
satisfy the constraint with substantial margin — the 250 m domain
radius always exceeds `5 × H_max` by at least 150 m.

---

## 7. CFD Integration Pipeline

### 7.1 Overview

The code to ingest CFD results, aggregate them onto the 10 m
morphometric grid, and combine directional simulations into an
annualised ensemble is implemented in `src/cfd_integration/` (1,000
lines across five modules) and covered by 46 passing unit tests.

**Input contract** (documented fully in
`src/cfd_integration/README.md`, provided to the CFD agent as the
delivery specification):

```
data/{site}/cfd_results/{patch_id}/{wind_direction}/
  sample_points.csv       # required — 15k rows at z = 1.5 m, 2 m spacing
  summary.json            # required — simulation metadata
  field.vtu               # optional — full 3D field
```

Plus `data/{site}/wind_rose.json` per site (now built from measured
INMET BDMEP / Iowa ASOS records; see §2.3 and Figure S5).

### 7.2 Module architecture

- **`schema.py`** — dataclasses for sample points, per-simulation
  metadata, campaign aggregates, wind roses; defines the 8 cardinal
  directions and runtime validators.
- **`io.py`** — CSV + JSON loading (primary path), optional VTU loading
  via `pyvista` (fallback), directory traversal for per-site campaigns.
- **`aggregate.py`** — spatial aggregation onto the morphometric grid.
  Primary function `aggregate_to_grid()` maps CFD samples onto individual
  10 m cells whose centroid falls inside the 100 m-diameter circular
  analysis patch (the scientifically defensible zone per Blocken 2015). `aggregate_to_domain()` is a
  supplementary function that includes the full 250 m radius for
  robustness checks.
- **`metrics.py`** — health-relevant scalar quantities:
  - `velocity_magnitude()` — √(U² + V² + W²)
  - `stagnation_fraction()` — fraction of samples with |U| < threshold
    (default 0.5 m/s, the commonly cited calm-air limit)
  - `turbulent_intensity()` — TI = √(2/3 · TKE) / U_ref
  - `ach()` — air change rate using the standard urban-climate
    formulation ACH = 3600 × ⟨|U|⟩ / L
  - `low_wind_percentile()` — worst-case stagnation (default p10)
  - `canyon_ventilation_efficiency()` — ⟨|U|⟩ / U_ref
- **`weighting.py`** — wind-rose-weighted combination of 8 directional
  simulations into an annualised metric. Supports three weighting modes:
  uniform (1/8 each), frequency-only (∝ f_dir), and frequency × speed
  (∝ f_dir · U_dir, which emphasises directions contributing more wind
  energy). `worst_case_direction()` finds the direction maximising any
  chosen metric (e.g. the direction producing most stagnant flow).

### 7.3 CFD agent specification

The full delivery contract (CSV schema, JSON schema, directory layout,
methodology requirements for turbulence model, boundary conditions,
meshing, post-processing, and convergence targets) is documented in
**`src/cfd_integration/README.md`**. Key decisions baked into the
contract:

- **Simulation domain:** cylindrical, 250 m radius, ≥ 6 × H_max height
  (≥ 90 m). Pedestrian sampling only valid within the central 100 m-
  diameter circular analysis patch.
- **Turbulence model:** k-ω SST (standard RANS for ABL flows at this
  scale).
- **Inlet:** logarithmic velocity profile, suburban roughness class
  (z₀ ≈ 0.03 m), turbulence intensity ≈ 0.15 at 10 m.
- **Simulations per site:** 8 cardinal directions, sharing one mesh
  with rotated inlet BC. 119 patches × 8 directions = 952 total
  simulations for the full campaign.
- **Sampling:** horizontal slice at z = 1.5 m, 2 m in-plane spacing,
  OpenFOAM `sample` utility, delivered as CSV.
- **Recommended validation patch:** **MAR-P07** (Maré, flat, 12.2 m
  max height, 1,200 buildings in domain) — the simplest end-to-end
  test case to validate the full pipeline before scaling.

### 7.4 End-to-end usage (post-CFD)

```python
from src.cfd_integration import (
    load_campaign_results, aggregate_to_grid, weighted_by_wind_rose,
)

campaign = load_campaign_results("vidigal")  # reads data/vidigal/cfd_results/
for patch_id in campaign.patch_ids():
    per_direction = {
        dir_: aggregate_to_patch(campaign.patches[patch_id][dir_],
                                 patch_center_xy)
        for dir_ in campaign.directions_for(patch_id)
    }
    annual = weighted_by_wind_rose(per_direction, campaign.wind_rose,
                                   weight_by="freq_speed")
    # annual = {annual_U_mean, annual_stagnation_frac, annual_ach, ...}
```

---

## 8. Repository Structure

Canonical output layout (as of 2026-04-13):

```
outputs/
├── {site}/
│   ├── morphometrics/         # SVF, per-building metrics, 10m grid,
│   │   ├── svf/              #   figures, report
│   │   ├── buildings/
│   │   ├── grid/
│   │   ├── figures/
│   │   └── report/
│   ├── morphometrics_20m/    # 20m grid (for resolution sensitivity)
│   └── sampling_cfd/
│       ├── pilot_sampling/   # 12–15 patches
│       └── campaign_sampling/ # 22–25 patches (pilot + new)
│
├── comparative/               # cross-site analysis
│   ├── pilot_summary/        # pilot-phase cross-site comparison
│   ├── final_allocation/     # 119-patch campaign summary
│   └── audits/               # topology audits
│
└── paper_figures/             # manuscript figures
    ├── fig_style.py          # shared style module
    ├── fig01_...py – figS4_...py
    ├── README.md
    └── exports/              # rendered PNG + SVG
```

Source modules:

```
src/
├── config.py                  # paths, CRS, filtering thresholds
├── cartography.py             # scale bars, north arrows, UTM formatting
├── morphometry/               # grid computation
│   ├── grid.py
│   ├── indicators.py
│   ├── audit.py
│   ├── figures.py
│   └── report.py
├── svf_v2/                    # SVF engine
│   ├── compute.py
│   ├── scene.py
│   ├── sampling.py
│   ├── paths.py              # per-site path registry
│   └── ...
├── solar/                     # solar access (phase 4)
└── cfd_integration/           # CFD ingestion (phase 7)
    ├── schema.py
    ├── io.py
    ├── aggregate.py
    ├── metrics.py
    ├── weighting.py
    └── README.md             # CFD agent specification
```

Key scripts:

| Script | Purpose |
|--------|---------|
| `run_morphometric_audit.py` | Per-site morphometric grid + SVF audit + report |
| `build_extended_context.py` | Site + city-wide building and DTM merger |
| `run_pilot_sampling.py` | 12-strata pilot batch (12–15 patches per site) |
| `run_campaign_sampling.py` | Campaign allocation (SVF-priority, target-total) |
| `build_wind_rose.py` | INMET → wind_rose.json ingestion |

---

## 9. Validation Summary

| Check | Status | Evidence |
|-------|--------|----------|
| Building geometry validity | PASS (10 repaired, all sites clean) | `outputs/comparative/audits/rocinha_topology_audit.md` |
| CRS consistency | PASS | all inputs verified EPSG:31983 on load |
| SVF passageway aggregation (no centroid artefacts) | PASS | diagnostic in `src/morphometry/grid.py::_aggregate_svf_to_grid` |
| DTM sentinel value handling | PASS (post-fix) | 15,286 sentinel pixels masked in Vidigal DTM |
| Extended context eligibility recovery | PASS | 55 % → 20 % domain exclusion at Vidigal |
| Blocken fetch compliance (all 119 patches) | PASS | minimum margin 150 m over 5 × H_max |
| Minimum inter-patch spacing (80 m) | PASS | realised minimum: 80–86 m across sites |
| Resolution sensitivity (10 m justified) | PASS | Figure S3; 10 m captures features 20 m loses |
| 12-strata coverage (≥ 1 site per stratum) | PASS | all 12 strata non-empty when pooled |
| CFD integration unit tests | PASS | 46 / 46 tests |

---

## 10. Known Limitations

1. **Wind forcing is placeholder.** All five `wind_rose.json` files
   carry `"quality_flag": "placeholder-prior"` and a per-site
   `expected_adjustment` note; the schema includes provenance fields
   (station id + coords, time window, observation count, calm
   fraction, anemometer height) but they are null pending ingestion of
   real data from the stations recommended in §2.3. The code path for
   ingestion (`scripts/build_wind_rose.py --inmet-csv`) is implemented
   and verified against the BDMEP CSV format; the outstanding step is
   downloading the yearly ZIPs and running it. Neutral stability is
   also assumed (see §2.3).

   Additionally, the earlier placeholder recommendations misattributed
   A652 ("Alto da Boa Vista") and A602 (Marambaia, for Complexo do
   Alemão). Both are corrected in §2.3's station table.

2. **CFD results not yet integrated.** `src/cfd_integration/` is
   tested but has not yet processed real simulation data. Any
   assumptions about sample-point density, column naming quirks, or
   edge cases in OpenFOAM output will only surface at first ingestion.
   Running the test suite against the first delivered patch
   (MAR-P07 recommended) will catch most issues.

3. **SVF validation against benchmark tools pending.** The Tregenza
   145-patch engine has been tested against synthetic canyons but not
   against an independent reference (e.g. SkyHelios, UMEP). For
   publication-grade claims about SVF absolute values, such a
   cross-validation should be added.

4. **Resolution sensitivity is 10 m vs 20 m only.** Finer grids
   (5 m, 2 m) would be prohibitively expensive at site scale but
   warrant a spot-check for the CFD patches specifically. Not a
   priority for the current milestone.

5. **Cidade de Deus excluded.** Data integrity issue leaves CDD
   outside the campaign. Adding it would require reprocessing the
   building footprints upstream of this repository.

---

## 11. Next Steps

**In this repository:**

- [ ] Replace placeholder wind roses with INMET station data (1–2 h
      each once CSVs are in hand)
- [ ] Optional: 20 m grids already exist; generate the variant Fig S3
      forms (`--variants` flag) if reviewers request the scatter or
      difference-map views
- [ ] Draft the Nature Cities manuscript (separate from this technical
      report)

**In the CFD repository:**

- [ ] Implement OpenFOAM case from per-patch exports (template in
      `outputs/{site}/sampling_cfd/campaign_sampling/patches/{id}/`)
- [ ] Validate pipeline on MAR-P07
- [ ] Mesh convergence study
- [ ] Run full 952-simulation campaign
- [ ] Post-process to `sample_points.csv` + `summary.json` per
      simulation

**After CFD results land here:**

- [ ] Verify first patch ingests correctly
- [ ] Annualise via wind-rose weighting
- [ ] Map CFD metrics onto the 10 m grid
- [ ] Extend Figure 5 to include wind-velocity panel
- [ ] Statistical analysis: SVF/λp/terrain as predictors of ACH and
      stagnation
- [ ] Health-outcome linkage

---

## Appendix A — Figure Index

| # | Title | File |
|---|-------|------|
| 1 | Study sites overview | `fig01_study_sites.png` |
| 2 | Morphometric distributions | `fig02_morphometric_distributions.png` |
| 3 | SVF–λp structural coupling | `fig03_svf_lambda_coupling.png` |
| 4 | CFD sampling design | `fig04_sampling_design.png` |
| 5 | Morphometric spatial maps | `fig05_morphometric_maps.png` |
| S1 | Correlation matrices (5 sites) | `figS1_correlation_matrices.png` |
| S2 | Extended context validation | `figS2_context_extension.png` |
| S3 | Resolution sensitivity (10m vs 20m) | `figS3_resolution_sensitivity.png` |
| — | Strata heatmap | `fig_strata_heatmap.png` |
| — | Candidate pool breakdown | `fig_candidate_pool.png` |
| — | Campaign allocation summary | `fig_campaign_allocation_summary.png` |
| — | Cross-site feature space | `fig_campaign_cross_site_featurespace.png` |
| — | All sites patch overview | `fig_all_sites_patches.png` |
| — | Patch metrics comparison | `fig_patch_metrics_comparison.png` |

All figures in `docs/technical_report/figures/`.

---

## Appendix B — Commit History (Relevant to CFD Campaign)

```
4dac8e8 feat: CFD integration module + Fig S3 resolution sensitivity
1890123 docs: update README and ROADMAP for v5.0.0 — CFD campaign complete
43f8554 feat: Nature Cities paper figures — 9 figures + shared style module
571ba47 feat: CFD sampling pipeline — stratified patch selection + campaign allocation
5e6e4c1 docs(figS3): canonicalize distribution-overlay as the main figure
9768e94 feat(wind-rose): scaffolding + template generator for CFD wind forcing
d477935 test(cfd-integration): 46 tests covering schema, IO, aggregation, metrics, weighting
```

---

## Appendix C — Key References

- Blocken, B. (2015). Computational Fluid Dynamics for urban physics:
  Importance, scales, possibilities, limitations and ten tips and
  tricks towards accurate and reliable simulations.
  *Building and Environment*, 91, 219–245.
- COST Action 732 (2007). Best practice guideline for the CFD
  simulation of flows in the urban environment.
- Stewart, I. D. & Oke, T. R. (2012). Local Climate Zones for Urban
  Temperature Studies. *Bulletin of the American Meteorological
  Society*, 93(12), 1879–1900.
- Tregenza, P. R. (1987). Subdivision of the sky hemisphere for
  luminance measurements. *Lighting Research & Technology*, 19(1),
  13–14.

---

*Prepared in `docs/technical_report/`. Available in both markdown
(`technical_report.md`) and PDF (`technical_report.pdf`) formats.*

**Regenerate the PDF:**
```bash
python docs/technical_report/build_pdf.py
```
*Pipeline: pandoc (markdown → HTML) then weasyprint (HTML → PDF).
Requires `pandoc` and the `weasyprint` Python package.*

**Regenerate all figures:**
```bash
for f in outputs/paper_figures/fig*.py; do python3 "$f"; done
```

**Regenerate campaign allocation:**
```bash
python scripts/run_campaign_sampling.py
```
