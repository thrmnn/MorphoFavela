# Changelog

All notable changes to the IVF (Informal settlements Vulnerability Framework)
project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once
a stable v1.0 is cut.

## [Unreleased]

### Added
- `scripts/download_inmet_zips.py` — robust INMET BDMEP yearly ZIP
  downloader with HTTP Range resume + per-attempt validation. Replaces
  the curl 7.68 + `-C -` workflow which trips on the INMET portal's
  habit of cutting large transfers mid-stream.
- `outputs/paper_figures/figS5_wind_roses.py` + the rendered PNG in
  `docs/technical_report/figures/` — 5-panel polar bar charts for the
  campaign sites.

### Changed
- All five `data/{site}/wind_rose.json` files now carry measured
  hourly observations across the full 2015–2024 window
  (`quality_flag: "measured"`); they previously held a Rio-coastal
  climatological prior (`placeholder-prior`).
- `scripts/build_wind_rose.py` — column-name normalisation strips
  Latin-1 accents (NFKD) so INMET headers like `direção_horaria`
  match ASCII candidates; date parsing normalises `/` → `-` so
  pre/post-2019 INMET CSVs (different date formats) parse together.
- Technical report §2.3 documents the measured roses and the
  station / window / n / calm-fraction table.
- README — adds a "Repository Map" section pointing at the CFD
  execution repo (`~/Airflow`), the wind-input pipeline, and the
  end-to-end ordering.

### Removed
- 8 stale planning / summary docs moved to `docs/archive/`
  (ALIGNMENT_EXPLANATION, DOCUMENTATION, GPU_SETUP,
  FAVELA_DATA_EXTRACTION_PLAN, FAVELA_EXTRACTION_SUMMARY,
  PYTORCH3D_GPU_IMPLEMENTATION, SVF_OPTIMIZATION_COMMIT_SUMMARY,
  TEST_SUITE_DESIGN).
- `src/patch_selection/` zombie directory (`__pycache__`-only
  leftover from the earlier tile-based pipeline removed in
  `0fcb3b6`).

## [0.1.0] — 2026-04-13

First milestone: complete CFD pre-flight stack.

### Added
- 119-patch stratified CFD sampling campaign across 5 favela sites
  (Vidigal 22, Rocinha 25, Rio das Pedras 22, C. do Alemão 25,
  Maré 25). 12-strata (SVF × slope × λp), 80 m greedy maximin
  spacing, 250 m circular CFD domain, 100 m-diameter circular
  analysis patch.
- `src/cfd_integration/` — dataclasses (`CFDSamplePoint`,
  `PatchSimulationMetadata`, `CFDPatchResult`, `WindRose`,
  `CFDCampaignResult`) + I/O + aggregation + metrics + wind-rose
  weighting, with a 46-test suite.
- Nature Cities paper figures: 9 publication-ready PNG/SVGs
  (Figs 1–5 + S1–S4) with shared style module (`fig_style.py`).
- Technical report (`docs/technical_report/technical_report.md` +
  `.pdf`) covering Sections 1–9 (sites, data, methods, sampling,
  CFD integration, validation plan, runbook).
- `CLAUDE.md` with project-specific workflow rules.
- Façade solar pipeline (PRs #8/#9): 3D ray-cast solar exposure on
  building façades with HTML report + dashboards.
- Morphometric audit pipeline: 12-indicator 10 m grid for each site
  (82,314 cells across 5 sites), publication figures, PDF report.
- Extended morphometrics: BCR, FAR, λp, λf, σH, slope, SVF, solar,
  porosity, density, Moran's I / LISA / Gi*.

### Notes
- CFD execution lives in a separate repo (`~/Airflow`); this repo
  produces the patch sampling and ingests results.
- Wind roses at this milestone were placeholder priors; replaced
  with measured INMET / Iowa ASOS data in the unreleased work above.
