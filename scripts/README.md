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
| `run_facade_solar.py` | `mf-facade-solar` | Per-storey façade solar exposure + WHO threshold compliance | `src.solar.facade` |
| `generate_facade_solar_report.py` | — | Interactive HTML dashboard from `run_facade_solar` output | (matplotlib + plotly) |
| `run_solar_animation.py` | — | Per-hour sunlit GeoPackage + manifest for dataviz animations. Reuses the observers from `svf_streets_solar.gpkg` (same geometry as the seasonal envelope) and appends one `lit_T{HHMM}` bool column per timestep. Writes to `outputs/{site}/dataviz/solar/`. | `src.solar.animation` |
| `compute_sectional_porosity.py` | — | Plan-view void fraction at z = 1.5 m as a wind-access proxy | (geopandas / shapely) |
| `compute_occupancy_density.py` | — | Built-volume / open-space ratio per analysis unit | `src.svf_v2.utils` |
| `classify_typology.py` | — | Settlement typology k-means clustering | `src.typology` |
| `plot_street_svf_distribution.py` | — | Histogram of street-level SVF for one or many sites | `src.svf_v2` |
| `plot_street_svf_with_isolines.py` | — | Street SVF overlaid on DTM hillshade + contours | `src.svf_v2` |

---

## Stage 3 — Per-feature combined indices + reports

| Script | CLI | Purpose | Library |
|---|---|---|---|
| `compute_deprivation_index_raster.py` | — | Continuous 2D raster of environmental deprivation (combines solar + SVF + porosity + occupancy) | `src.exposure`, `src.solar` |
| `run_morphometric_audit.py` | `mf-morphometry` | One-shot per-site audit: figures + PDF report | `src.morphometry` |
| `compare_areas.py` | `mf-compare` | Formal-vs-informal comparison report (statistical tests + PDF) | `src.metrics`, scipy.stats |
| `generate_report.py` | — | Single-area or comparative PDF report | `src.morphometry.report` |

---

## Stage 4 — CFD patch sampling

| Script | CLI | Purpose | Library |
|---|---|---|---|
| `run_pilot_sampling.py` | `mf-pilot-sampling` | Stratified 12-strata pilot batch (12-15 patches per site) | `src.morphometry` (sampling logic in script) |
| `run_campaign_sampling.py` | `mf-campaign-sampling` | Incremental top-up to 22-25 patches per site (SVF-priority) | (sampling logic in script) |

CFD execution itself happens in the separate `~/Airflow` repo — see
the top-level `README.md` Repository Map.

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
