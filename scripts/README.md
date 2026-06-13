# `scripts/`

Executable entry points for the MorphoFavela pipeline. Library code lives in
`src/`; everything in this directory wraps a function in `src/`
behind argparse.

After `pip install -e .` the high-traffic scripts are also exposed as
`mf-*` console commands. Both forms below are equivalent:

```bash
python scripts/run_svf_v2.py --area vidigal   # always works
mf-svf --area vidigal                         # after pip install -e .
```

The console-script aliases registered in `pyproject.toml` are
listed in the table headers below (in the **CLI** column where
applicable).

## How to read this index

Scripts are grouped by stage of the pipeline (the same order as the
walkthrough in the top-level [`README.md`](../README.md)). Each
entry lists:

- **Purpose** — one line.
- **Library backing** — which `src/` module does the real work.
- **Typical invocation** — short example.

When you add a new entry-point script, add a row here.

---

## Stage 1 — Per-site context preparation

| Script | CLI | Purpose | Library |
|---|---|---|---|
| `build_extended_context.py` | `mf-context` | Clip city-wide buildings + DTM to favela boundary + 300 m buffer | `src/svf_v2/paths` |
| `download_inmet_zips.py` | `mf-download-inmet` | Robust resumable downloader for INMET BDMEP yearly archives | (urllib only) |
| `extract_inmet_stations.py` | `mf-extract-inmet` | Pull per-station CSVs out of yearly INMET ZIPs and concatenate | (zipfile / pandas only) |
| `build_wind_rose.py` | `mf-wind-rose` | Build `data/{site}/wind_rose.json` from INMET CSV or Iowa ASOS METAR | `src.cfd_integration.schema` |

```bash
# Wind input pipeline (one-off, shared by all sites):
python scripts/download_inmet_zips.py --years 2015..2024 --out-dir data/inmet/raw
python scripts/extract_inmet_stations.py --zips-dir data/inmet/raw --out-dir data/inmet/processed --stations A652 A636 A621 A602
python scripts/build_wind_rose.py --site vidigal --inmet-csv data/inmet/processed/concat/A652_2015_2024.csv
```

---

## Stage 2 — Per-feature morphometric analyses

These are **independent** — run in any order or parallel.

| Script | CLI | Purpose | Library |
|---|---|---|---|
| `compute_urban_morphology.py` | — | Zone-level metrics: BCR, FAR, plot ratio, frontal area | `src.urban_morphology` |
| `run_svf_v2.py` | `mf-svf` | Sky View Factor (GPU-capable) on a 2 D ground grid | `src.svf_v2` |
| `compute_solar_access.py` | `mf-solar` | Hours of direct sun on the winter-solstice ground grid | `src.solar` |
| `run_street_solar.py` | — | Per-observer seasonal solar access (4 reference dates + annual aggregate). Drives off `svf_streets.gpkg`, writes `svf_streets_solar.gpkg`. | `src.solar.seasonal` |
| `run_aspect_analysis.py` | — | Per-cell aspect / slope summary for sloped-terrain solar dissociation analysis | `src.svf_v2`, `numpy` |
| `run_facade_solar.py` | `mf-facade-solar` | Per-storey façade solar exposure + WHO threshold compliance | `src.solar.facade` |
| `generate_facade_solar_report.py` | — | Interactive HTML dashboard from `run_facade_solar` output | (matplotlib + plotly) |
| `run_solar_animation.py` | — | Per-hour sunlit GeoPackage + manifest for dataviz animations. Reuses the observers from `svf_streets_solar.gpkg` (same geometry as the seasonal envelope) and appends one `lit_T{HHMM}` bool column per timestep. Writes to `outputs/{site}/dataviz/solar/`. | `src.solar.animation` |
| `compute_sectional_porosity.py` | — | Plan-view void fraction at z = 1.5 m as a wind-access proxy | (geopandas / shapely) |
| `compute_occupancy_density.py` | — | Built-volume / open-space ratio per analysis unit | `src.svf_v2.utils` |
| `classify_typology.py` | — | Settlement typology k-means clustering | `src.typology` |
| `plot_street_svf_distribution.py` | — | Histogram of street-level SVF for one or many sites | `src.svf_v2` |
| `plot_street_svf_with_isolines.py` | — | Street SVF overlaid on DTM hillshade + contours | `src.svf_v2` |
| `validate_svf_against_umep.py` | — | Per-site cross-validation of MorphoFavela's ray-cast SVF against UMEP's shadow-cast `svfForProcessing153`. Writes `outputs/{site}/morphometrics/svf/umep_validation/`. | `src.svf_v2`, UMEP |

> **⚠ SVF output path — producer/consumer split (mandatory ordering).**
> `run_svf_v2.py` writes its raw outputs to **`outputs/{site}/svf_v2/`**
> (`svf_streets.gpkg`, `svf_grid.*`, `scene.stl`, figures). But every
> downstream consumer — `run_street_solar.py`, `compute_cross_site_stats.py`,
> `run_aspect_analysis.py`, `run_diagnostic_models.py`,
> `run_predictor_analysis.py`, `build_street_observers.py`,
> `compare_mingze_vidigal.py` — reads the **canonical
> `outputs/{site}/morphometrics/svf/`** tree. The bridge is
> `run_morphometric_audit.py`, which copies `svf_v2/*` → `morphometrics/svf/`
> (overwriting, since the 2026-06 copy-guard fix). **So after re-running SVF
> you must run `run_morphometric_audit.py` (or copy the files yourself)
> before any consumer, or they will silently read stale geometry.** This
> split is the mechanism behind the 2026 street-SVF drift; until a future
> change makes `run_svf_v2` write `morphometrics/svf/` directly, treat
> `run_svf_v2 → run_morphometric_audit → consumers` as a hard ordering.

---

## Stage 3 — Per-feature combined indices + reports

| Script | CLI | Purpose | Library |
|---|---|---|---|
| `compute_deprivation_index_raster.py` | — | Continuous 2D raster of environmental deprivation (combines solar + SVF + porosity + occupancy) | `src.exposure`, `src.solar` |
| `run_morphometric_audit.py` | `mf-morphometry` | One-shot per-site audit: figures + PDF report | `src.morphometry` |
| `compare_areas.py` | `mf-compare` | Formal-vs-informal comparison report (statistical tests + PDF) | `src.metrics`, scipy.stats |
| `generate_report.py` | — | Single-area or comparative PDF report (segment SVF now length-weighted per audit C1, fix d8175ca) | `src.morphometry.report` |
| `compute_cross_site_stats.py` | — | Aggregated cross-site distributional stats for paper figures and report tables | `src.metrics` |
| `run_predictor_analysis.py` | — | Multi-predictor regression / changepoint analysis for SVF → wind / solar dissociation | `src.metrics`, statsmodels |
| `run_diagnostic_models.py` | — | Fits the four-state diagnostic taxonomy (λf vs U_mean) used in BRISA Fig 04 | `src.svf_v2`, `src.cfd_integration` |

---

## Stage 4 — CFD patch sampling

| Script | CLI | Purpose | Library |
|---|---|---|---|
| `run_pilot_sampling.py` | `mf-pilot-sampling` | Stratified 12-strata pilot batch (12-15 patches per site) | `src.morphometry` (sampling logic in script) |
| `run_campaign_sampling.py` | `mf-campaign-sampling` | Incremental top-up to 22-25 patches per site (SVF-priority) | (sampling logic in script) |
| `select_pilot_candidates.py` | — | Filter / rank candidate pilot patches against eligibility criteria | (sampling logic in script) |
| `audit_rectangular_domain.py` | — | Verify per-patch rectangular-domain compatibility before CFD submission | `src.cfd_integration` |
| `analyze_cfd_results.py` | — | Post-CFD per-patch summary (residuals, U_mean fields, convergence) once results return from `~/Airflow` | `src.cfd_integration` |
| `generate_synthetic_cfd_results.py` | — | Gitignored synthetic CFD tree for testing the analyzer without HPC runs (see memory `project-vdgp02-synthetic-cfd`) | `src.cfd_integration` |
| `migrate_indicators_rectangular_v1.py` | — | One-shot migration script for the rectangular-domain indicator schema bump (run once after pulling a v0 → v1 patch) | `src.cfd_integration` |

CFD execution itself happens in the separate `~/Airflow` repo — see
the top-level `README.md` Repository Map.

---

## Stage 5 — Distribution bundles, dashboards, diagnostics

These wrap analysis outputs into shareable artefacts (per-site
dashboards, data bundles for collaborators) and produce one-off
diagnostic figures. They consume the artefacts that stages 1-4
produce; they never re-cast rays or re-run the solver.

| Script | CLI | Purpose | Library |
|---|---|---|---|
| `build_street_observers.py` | — | Canonical per-site street-observer network → `outputs/{site}/sampling_streets/observers.{gpkg,geojson,parquet}` + manifest, ASCII-rename maré→mare for cross-platform safety. Also `--bundle` stages a versioned drop under `outputs/_distribution/street_observers_v1/`. | `src.svf_v2.sampling`, `src.svf_v2.paths` |
| `build_site_dashboard.py` | — | *Folha de Rua* static A3 site sheet (PNG + PDF + 8 atomic panel PNGs) under `outputs/_distribution/site_dashboards/{site}/`. Uses length-weighted segment SVF. | `src.viz`, matplotlib |
| `build_html_dashboard.py` | — | Interactive HTML site dashboard (Leaflet observer map + Plotly histogram + 2D-density scatter + methodology drawer with audit anchors). Output: `outputs/_distribution/html_dashboards/{site}/index.html`. | (Leaflet + Plotly via CDN) |
| `build_mingze_bundle.py` | — | 3D-data bundle for the Ladybug solar-access collaborator (DTM + 3D footprints with `altura` + boundary + canonical observers, per site). Writes tarball + sha256 under `outputs/_distribution/mingze_3d_bundle_v1/`. | (shutil / hashlib only) |
| `build_diagnostic_map.py` | — | Per-site SVF / solar / observer-density diagnostic map | `src.svf_v2`, matplotlib |
| `build_vidigal_diagnostic_map.py` | — | Specialised Vidigal diagnostic map for the BRISA paper | `src.viz`, matplotlib |
| `compare_mingze_vidigal.py` | — | Accuracy comparison of our Vidigal winter-solstice solar hours against Mingze's Ladybug run (paired 1-to-1 by row order, n = 6876). MAE / RMSE / bias / Pearson, Bland-Altman, residual map + per-observer GPKG. Output: `outputs/comparative/vidigal_vs_mingze/`. | (pandas / geopandas / scipy) |

---

## Subdirectories

- `scripts/hpc/` — small SLURM helpers; the bulk of the HPC code is
  in the `~/Airflow` repo (mesh, solve, postprocess, submit).
- `scripts/data_utils/` — small shared loaders for repeated ad-hoc
  ingestion.
- `scripts/debug/` — one-off diagnostics that aren't part of the
  pipeline.
- `scripts/shell/` — bash one-liners (deprecated; kept for
  archaeology).

---

## Notes on cleanup history

The street-level SVF + solar + sky-exposure scripts
(`compute_solar_access_streets.py`, `analyze_sky_exposure_streets.py`,
`compute_deprivation_streets.py`) were removed on 2026-04-29
because they had been broken since the `src/svf_v2/` refactor
— they imported sibling scripts that were deleted at that time.
The library functions remain available:
`src.solar.compute.compute_solar_access_streets` and
`src.svf_v2.sampling.sample_street_points`. Anyone wanting
street-level analysis can write a thin CLI wrapper using those
helpers in <100 lines; the deleted scripts are preserved in git
history if their implementations are useful as a reference.

The unit-level deprivation script (`compute_deprivation_index.py`)
was removed on 2026-05-03 alongside the legacy `run_area_analyses.py`
orchestrator. Only `compute_deprivation_index_raster.py` remains; its
core formulas live in `src/exposure/deprivation.py` and unit-level
aggregation is achievable by passing analysis-unit polygons via the
`--units` flag.
