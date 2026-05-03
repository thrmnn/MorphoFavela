# Changelog

All notable changes to the IVF (Informal settlements Vulnerability Framework)
project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once
a stable v1.0 is cut.

## [Unreleased]

### Changed — production-readiness cleanup (May 2026, two passes)

- **Pass 1** (`09427fd`, 2026-05-03): −7,351 lines across 39 files.
  Deleted 26 of 27 docs in `docs/archive/` (the 27th, `Morphometrics.md`,
  was the only document with lasting reference value and moved to
  `docs/methodology/morphometric_indicators.md`); 7 ad-hoc design docs
  in `tests/`; the orphan `feature/gpu-svf-acceleration/` stub README;
  the empty `tests/test_patch_selection/` directory left over from the
  April 2026 source deletion; and `scripts/analyze_morphology_risk.py`
  (zero callers, no `pyproject` entry, no docs reference). Created
  `docs/methodology/` and moved top-level `SKY_EXPOSURE_METHODOLOGY.md`
  + `STREET_SVF_USAGE.md` into it.
- **Pass 2** (`c2808c2`, 2026-05-03): −1,605 lines across 9 files.
  Deleted the legacy Phase-2 orchestrator trio
  (`scripts/run_area_analyses.py`, `scripts/calculate_morphology_metrics.py`,
  `scripts/compute_deprivation_index.py`) — all wrote to the deprecated
  `outputs/{site}/{svf,solar,porosity,density}/` layout that predated the
  canonical `morphometrics/`/`sampling_cfd/`/`comparative/` layout;
  `requirements.txt` (deps now declared exclusively in `pyproject.toml`,
  with `joblib>=1.3.0` added since it was previously declared only in
  `requirements.txt`); two stale `docs/guides/*` mini-tutorials that
  referenced the deleted orchestrator + non-campaign sites
  (`vidigal_tls`, `copacabana`).
- Test suite collection unchanged across both passes (577 tests, 508 in
  the non-integration set). All deletions remain accessible via git
  history; no behaviour change.
- Source: [`docs/PRODUCTION_READINESS_PLAN.md`](docs/PRODUCTION_READINESS_PLAN.md)
  Track A (cleanup), executed against Wave 1 audit results from five
  parallel `Explore` agents.

### Added — `numerical-claims-auditor` subagent

- Validator-class subagent at
  [`.claude/agents/numerical-claims-auditor.md`](.claude/agents/numerical-claims-auditor.md)
  that extracts every numerical claim (counts, percentages, ranges,
  comparators, summary statistics, dates) from a markdown document
  (default: `docs/technical_report/technical_report.md`) and verifies
  each against a traceable source — `campaign_patches.csv`,
  `patch_meta.json` fields, `summary_stats.csv`, `wind_rose.json`, etc.
  Reports per-claim VERIFIED / MISMATCH / UNVERIFIABLE. Read-only.
- Targets the §6.5 Blocken-class bug: prose drift from the underlying
  data when the constraint check itself is correct. A scripted check
  cannot replicate this — it requires reading prose, identifying a
  numerical claim, finding its source, and comparing. Recurring
  trigger: any TR edit that touches numbers, any sampling/grid
  regeneration, any pre-release sweep.
- Brings the project subagent team to seven (4 validators + 3 workflow
  accelerators). Loaded at session start, so requires Claude Code
  restart before first use.

### Fixed — §6.5 Blocken margin claim corrected

- Technical report §6.5 previously stated the 250 m CFD domain radius
  "always exceeds 5 × H_max by at least 150 m". An audit of all 119
  `patch_meta.json` files showed the actual minimum margin was 114 m
  (RDP-P15, `H_max_analysis` = 27.3 m), with 11 of 119 patches under
  150 m — all at Rio das Pedras (5/22) or Rocinha (6/25), the two
  sites with the tallest analysis-patch structures. The Blocken
  constraint itself (`5 × H_max ≤ 250 m`) holds at every patch
  (`blocken_ok = true` in every `patch_meta.json`); only the
  descriptive prose around the margin had drifted. Rewritten with
  the actual range (114–215 m, median 180 m) and a per-site
  breakdown.
- This drift is the canonical bug class the new
  `numerical-claims-auditor` agent targets. PDF rebuilt in the
  same commit (`53396db`).

### Added — SVF cross-validated against UMEP (closes §10.3 limitation)

- `scripts/validate_svf_against_umep.py` benchmarks the IVF Tregenza-145
  ray-cast SVF against UMEP's shadow-casting SVF
  (`svfForProcessing153`, Lindberg & Holmer 2010, used in SOLWEIG) on a
  rasterised DSM. The two engines are algorithmically distinct
  (mesh ray-casting vs raster shadow-casting); their agreement is the
  evidence reviewers want for "SVF absolute values are defensible".
- UMEP source is auto-fetched on first run into `vendor/umep_processing/`
  (gitignored) and patched to bypass its QGIS-only `util/__init__.py`.
  No QGIS install needed; only `gdal` (and a `numba`-compatible
  numpy < 2.3, both pulled into the conda env).
- Five-site benchmark at 1 m DSM, height-matched at z = 1.5 m
  (`--observer-height` flag, building heights lowered by `h` before
  passing to UMEP — equivalent to lifting the integration plane up by
  `h`). Vidigal + Maré completed 2026-04-30; Rocinha + Rio das Pedras
  + Complexo do Alemão added 2026-05-01:

  | Site | n | r² | slope | RMSE | bias |
  |---|---:|---:|---:|---:|---:|
  | Vidigal | 2,510 | 0.68 | 0.96 | 0.12 | +0.01 |
  | Rocinha | 5,646 | 0.81 | 1.01 | 0.14 | +0.09 |
  | Rio das Pedras | 1,503 | 0.93 | 1.25 | 0.11 | +0.08 |
  | Complexo do Alemão | 4,838 | 0.76 | 0.92 | 0.17 | +0.14 |
  | Maré | 9,516 | 0.94 | 0.97 | 0.12 | +0.09 |

  Four of five regression slopes (Vidigal, Rocinha, Complexo do Alemão,
  Maré) are within ±8 % of unity. Rio das Pedras at 1.25 reflects the
  smallest valid grid (n = 1,503) concentrated in the 0.3–0.5 SVF band
  where the 153-patch shadow-cast and 145-patch ray-cast integrators
  diverge most. r² (0.68–0.94) tracks site relief: low-relief fabrics
  (Maré 0.94, Rio das Pedras 0.93) have the strongest per-cell
  agreement; high-relief hillside settlements (Vidigal 0.68,
  Complexo do Alemão 0.76, Rocinha 0.81) produce more cell-to-cell
  variation between integrators. All biases are positive (+0.01 to
  +0.14), consistent with sub-1.5 m structures being zeroed in IVF's
  height-shifted input but retained in UMEP. UMEP is aggregated to the
  10 m grid only over non-building pixels — without that mask the
  comparison is contaminated by rooftop pixels that UMEP scores ~1.0
  but IVF never samples. The mask brought Vidigal r² from 0.04 to 0.68.
- Outputs land in `outputs/{site}/morphometrics/svf/umep_validation/`
  (per-cell scatter PNG, per-cell CSV, summary stats CSV). Scatters
  are copied to `docs/technical_report/figures/figS{6,7,8,9,10}*.png`
  for the report.
- Technical report: §4 SVF definition gains a 5-site cross-validation
  table and inter-site interpretation prose; §10.3 "SVF validation
  pending" → "SVF validated against UMEP across all 5 sites
  (limitation closed 2026-04-29 for Vidigal, height-matched 2026-04-30,
  extended to all 5 sites 2026-05-01)"; figure index gains figS6 (Vidigal),
  figS7 (Maré), figS8 (Rocinha), figS9 (Rio das Pedras), figS10 (Complexo
  do Alemão); §11 next-steps box for UMEP cross-val ticked. PDF rebuilt.

### Added — report-sync hook: figure-without-md WARN

- `.claude/hooks/check_report_sync.py` gains a soft rule: if any
  `docs/technical_report/figures/*.{png,pdf,svg}` is staged without
  `technical_report.md`, emit a WARN. Catches the "regenerated
  figure + stale report numbers" pattern earlier than the existing
  `.md ↔ .pdf` pairing rule. Stays advisory (exit 0) — cosmetic
  figure tweaks shouldn't force a report touch.

### Fixed — synthetic CFD generator now self-validates against ingestor

- `scripts/generate_synthetic_cfd_results.py` previously stored a
  `U_mag` computed from the pre-perturbation magnitude while U/V/W
  components had independent perturbations added on top, leaving
  `|U_mag - sqrt(U² + V² + W²)| ≈ 0.10–0.15` on every row.
  `cfd-results-ingestor` flagged 160/160 patch-directions as WARN
  for the inconsistency. Fixed by deriving `U_mag` from the perturbed
  components — now self-validates at machine precision.
- Default `--n-samples` bumped from 5,000 to 15,000 to match the
  validator's expected ~15k/patch for a 250 m domain at 2 m grid.
  Tests pass `n_samples_per_direction` explicitly so they're
  unaffected (63 CFD tests still green).
- With both fixes, future synthetic runs hit PASS, leaving WARN/FAIL
  exclusively as a producer-drift signal when real Airflow results
  land.

### Fixed — `src/exposure` package vs module collision (regression from `ede48ad`)

- The deprivation cleanup in `ede48ad` created `src/exposure/` as a
  package without removing the original `src/exposure.py` single-file
  module. Python's import machinery picked up the package and shadowed
  the original module's symbols (`compute_exposure_index`,
  `compute_zone_solar_deficit`, `plot_exposure_panel`, …), breaking
  `from src.exposure import compute_exposure_index, ...` everywhere.
  Compounding this, `tests/test_exposure.py` (the file) and
  `tests/test_exposure/` (the new directory holding `test_deprivation.py`)
  collided in pytest collection, taking the full suite from green to
  failing-on-collection.
- `src/exposure.py` → `src/exposure/sky_exposure.py`; package
  `__init__.py` re-exports the original symbols alongside the
  deprivation formulas. `tests/test_exposure.py` →
  `tests/test_exposure/test_sky_exposure.py` so both test modules
  live under the same package and pytest can collect both.
- Full suite now: **526 passed, 22 skipped** (was: collection error).

### Added — result-side CFD analysis pipeline (`scripts/analyze_cfd_results.py`)

- New `scripts/analyze_cfd_results.py` orchestrates the full chain
  for one site: discover returned patches → aggregate per-direction
  → wind-rose annualise (frequency × speed weighting by default) →
  per-patch indicator table joined with morphometric covariates →
  per-cell cell-level annualisation onto the 10 m grid → within-site
  OLS predictor regression (ACH/U_mean ~ SVF + λp + slope + σ_h) →
  Figure 5 wind-panel PNG. Auto-detects IVF-native CSV and
  Airflow-native parquet layouts via the existing
  `src.cfd_integration.io` adapter.
- New `scripts/generate_synthetic_cfd_results.py` produces a complete
  `data/{site}/cfd_results/` tree (per-patch × 8-direction sample
  files + summary.json) keyed off `campaign_patches.csv`. Mean
  in-canopy U_mag is modulated by patch SVF and λp so the predictor
  regression has the expected sign structure (lower SVF + higher λp
  → lower ventilation). Both layouts (`csv`, `parquet`) produced
  identically; `--n-patches`, `--n-samples`, `--seed` for control.
- Outputs written to `outputs/{site}/cfd_analysis/`:
  `per_patch_indicators.csv`, `grid_with_cfd.gpkg`,
  `predictor_regression.csv`, `figures/fig5_wind_panel.png`,
  `coverage.json` (which patches/directions returned, what was
  missing, weighting method).
- Smoke-tested end-to-end on real Vidigal campaign patches (5/22
  patches × 8 directions × 2,000 samples → full chain runs in 2.7 s,
  produces 386 covered grid cells, U_mean range 0.63–2.38 m/s).
- 6 new tests in `tests/test_cfd_integration/test_analyze.py` cover
  the synthetic generator (CSV + parquet layouts), the full chain,
  partial coverage (one direction missing), patch absence (entire
  patch missing), and parquet auto-detection. Full CFD test suite
  stays green at 63 tests.
- Two new entry points: `ivf-synthetic-cfd` and `ivf-analyze-cfd`.
- Docs/§7.4 update is deferred until real VDG-P07 results land —
  the synthetic chain validates the plumbing but the manuscript
  text wants real coefficients to cite.

### Changed — technical report §10/§11 reconciled with shipped reality

- §10.1 rewritten: "Wind forcing is placeholder" → "Neutral stability
  is assumed". All five `wind_rose.json` files have carried
  `quality_flag: "measured"` since 2026-04-27 with the full 2015–2024
  window (n = 64,088–89,439 hourly records); the methodological
  caveat that remains is the neutral-stability assumption, not the
  data state. The historical placeholder framing was a 3-day-old
  contradiction in the project's primary deliverable document.
- §10.5 (Cidade de Deus) firmed up: campaign locked to 5 sites,
  re-onboarding CDD is out of scope for the current cycle.
- §11 "Next Steps" updated: wind ingestion checked off; pilot patch
  reference changed from MAR-P07 to VDG-P07 (the actual one in flight
  at MIT ORCD); added explicit step for the result-side analysis
  pipeline; CFD-results checklist annotated with the agent
  (`cfd-results-ingestor`) and weighting module that handle each item.
- §2.3 station-table footnote corrected: METAR ingestion via
  `build_wind_rose.from_iowa_asos_csv` is implemented, not "not yet
  implemented".
- PDF rebuilt (`docs/technical_report/technical_report.pdf`).

### Removed — broken `_streets.py` scripts (cleanup)

- `scripts/compute_solar_access_streets.py`,
  `scripts/analyze_sky_exposure_streets.py`,
  `scripts/compute_deprivation_streets.py` deleted. They had been
  broken since the `src/svf_v2/` refactor — they imported
  `scripts.compute_svf_streets` and `scripts.analyze_sky_exposure`
  which no longer exist. Library functions for street-level analysis
  remain available (`src.solar.compute.compute_solar_access_streets`,
  `src.svf_v2.sampling.sample_street_points`); anyone needing the
  CLI form can write a thin wrapper from those primitives. Old
  implementations preserved at commit `82f7f44~1` for reference.
- `scripts/README.md` "Known issues" section rewritten to reflect
  the cleanup (no more dangling broken-script row).

### Added — `src/exposure/` shared deprivation formulas

- New `src/exposure/deprivation.py` module hosts the three formulas
  shared between `compute_deprivation_index.py` (unit-level) and
  `compute_deprivation_index_raster.py` (raster-level):
  `solar_deficit`, `ventilation_deficit`, `hotspot_index`. The
  formulas are deliberately type-agnostic — they work on numpy
  ndarrays, pandas Series, and DataFrame columns alike via
  duck-typed arithmetic + `.clip()`.
- Both deprivation scripts patched to import from the shared module
  instead of inlining the formulas. The scripts continue to handle
  type-specific concerns (numpy NaN propagation, pandas percentile
  ranking, plotting) themselves; only the math is centralised.
- 16 new tests in `tests/test_exposure/test_deprivation.py`
  verifying numpy/pandas equivalence and edge cases (zero hours,
  exactly-at-reference, above-reference clipping). All pass.
- Full unit↔raster consolidation onto a `--resolution {unit,raster}`
  flag is still pending — but the residual duplication after this
  pass is mostly the I/O and plotting wrappers, not the math, so
  the two scripts can no longer silently disagree on what the index
  *means*.

### Added — Airflow result adapter + report-sync hook

- `src/cfd_integration/io.py::load_patch_parquet` — parquet reader
  parallel to `load_patch_csv`. Single-file or multi-file (concatenated
  row-wise, used when OpenFOAM emits one parquet per processor
  decomposition).
- `src/cfd_integration/io.py::_normalize_wind_direction` — accepts
  both the IVF-native cardinal form (`N`, `NE`, …) and the Airflow-
  native `wind_NNN` form (`wind_000` → `N`, `wind_045` → `NE`, …,
  `wind_315` → `NW`); off-axis degrees return `None` so callers can
  warn-and-skip.
- `load_campaign_results` updated to auto-detect both layouts per
  direction (CSV vs. parquet) and dispatch transparently — including
  the mixed case where one direction is CSV and another is parquet
  within the same patch.
- 12 new tests in `tests/test_cfd_integration/test_schema_io.py`
  covering direction normalisation, single-file parquet round-trip,
  multi-parquet concat, U_mag auto-compute, and three campaign-loader
  layout scenarios (IVF native, Airflow native, mixed). All 57 CFD
  tests pass; full fast-mark suite stays green at 81 tests.
- `.claude/hooks/check_report_sync.py` + `.claude/settings.json` —
  PreToolUse hook on Bash that fires on `git commit` and surfaces
  a punch list of report-sync findings: hard FAIL on `.md` ↔ `.pdf`
  mismatch and on paper-figure script changes without a matching PNG
  copy; advisory-only on triggers that need LLM judgment (was that
  `scripts/X.py` change pipeline-relevant?). Always exits 0
  (advisory mode); flip to `exit 2` to make it blocking once the
  false-positive rate is characterised. The full LLM-backed
  `report-sync-auditor` agent remains available for explicit
  invocation when the hook flags advisories.

### Changed — Airflow result adapter

- `src/cfd_integration/README.md` and `data/README.md` document both
  on-disk layouts as accepted; the `summary.json` `wind_direction`
  field still uses the cardinal form regardless of which directory
  layout the producer used.
- `.claude/agents/cfd-results-ingestor.md` no longer treats the
  Airflow `wind_NNN/*.parquet` layout as "drift" — both layouts are
  PASS; only unknown / off-axis directories and missing files remain
  FAIL. The Bash recipes in the agent prompt updated to use
  `load_campaign_results` (auto-detect) instead of inferring layout.
- `.gitignore` exception expanded from `.claude/agents/` to also
  cover `.claude/hooks/` and `.claude/settings.json`. Per-user
  `settings.local.json` and `projects/` session data stay ignored.

### Added — agent team

- Six project-scoped Claude Code subagents under `.claude/agents/`,
  split into two classes:
  - **Validators** (read-only, report-only):
    `data-contract-checker` (verifies `data/{site}/` against
    `data/README.md` schema + measured-quality gate),
    `sampling-auditor` (audits CFD patch sampling against
    per-site counts, stratum `n_target` coverage, 80 m maximin
    spacing, per-patch integrity), `report-sync-auditor` (maps a
    git diff to the `CLAUDE.md` "Technical report" triggers and
    flags `.md` ↔ `.pdf` ↔ `figures/` drift).
  - **Workflow accelerators** (orchestrate multi-step work, stop
    at manual steps): `site-onboarder` (the 7-step new-site
    checklist; halts at the deliberately-manual DTM-clip step),
    `wind-ingestion` (INMET/ASOS download → extract → build rose,
    encoding the 3 known INMET quirks: server cuts, post-2019
    date format, accent-bearing column names),
    `cfd-results-ingestor` (validates returns from `~/Airflow`
    against the `src/cfd_integration/` contract and flags the
    `wind_NNN/*.parquet` → `wind_{N..NW}/sample_points.csv`
    producer drift).
- Trial run of `sampling-auditor` against the current 119-patch
  state cleanly surfaced the previously-documented Rio das Pedras
  shortfall (stratum `SVF2_SLP2_LP2`: n_target=1, 0 placed) — that
  gap was already accepted and documented in technical report §6.4
  as a genuine morphological scarcity (only 7 eligible cells, all
  within 80 m of pilot patches; sampler can't satisfy maximin).
  Added `docs/cfd_sampling_overrides.yaml` (tracked) so the agent
  downgrades documented gaps from FAIL → WARN — keeps the FAIL
  signal reserved for genuinely-new misses while keeping the
  accepted gap visible on every run.
- `.claude/agents/README.md` documents the agent design rules
  (read-only validators, idempotent accelerators, stop loudly at
  manual steps, never push or commit, cite the contract).
- `CONTRIBUTING.md` "Project subagents" section points contributors
  at the team and notes that agents are loaded at session startup.

### Changed — agent team

- `.gitignore` reworked from `.claude/` (full-directory ignore) to
  `.claude/*` + `!.claude/agents/`, so project agents track but
  session data and local settings stay ignored. (Negation only
  works when the parent directory itself is not excluded.)

### Added — production-grade pass

- `CHANGELOG.md`, `CONTRIBUTING.md`, `CITATION.cff` (this file is
  what activates GitHub's "Cite this repository" widget).
- `.pre-commit-config.yaml` — ruff format + check + standard
  pre-commit-hooks (trailing whitespace, EOF, YAML/TOML syntax,
  merge-conflict markers, large-file guard at 2 MB).
- `data/README.md` (now tracked) — input contract per `data/{site}/`
  + INMET / Iowa ASOS recovery commands + manual-clip steps.
- Per-package READMEs across `src/`: `svf_v2/`, `solar/`,
  `morphometry/`, `visualization/` (matching the existing
  `cfd_integration/README.md` standard).
- `scripts/README.md` — index of all 27 entry-point scripts grouped
  by pipeline stage, with a "CLI" column showing the `ivf-*`
  console aliases.
- 11 `ivf-*` console scripts via `[project.scripts]` in
  `pyproject.toml` — `ivf-svf`, `ivf-morphometry`, `ivf-wind-rose`,
  etc. Available after `pip install -e .`.
- `[tool.ruff]` configuration in `pyproject.toml` with documented
  pragmatic ignores for cosmetic checks; CI now lints `scripts/`
  and `outputs/paper_figures/` in addition to `src/`+`tests/`.

### Changed — production-grade pass

- README rewritten: 611 → 279 lines, single canonical walkthrough
  (clone → install → run on vidigal end-to-end → paper figures)
  replacing four overlapping Quick Start / Usage Examples /
  Configuration / Output Files blocks. Per-module details now
  defer to `src/<module>/README.md`.
- 8 stale planning docs moved to `docs/archive/`
  (ALIGNMENT_EXPLANATION, DOCUMENTATION, GPU_SETUP,
  FAVELA_DATA_EXTRACTION_PLAN, FAVELA_EXTRACTION_SUMMARY,
  PYTORCH3D_GPU_IMPLEMENTATION, SVF_OPTIMIZATION_COMMIT_SUMMARY,
  TEST_SUITE_DESIGN).
- `.gitignore` tightened (build/, dist/, .mypy_cache/, .coverage,
  htmlcov/) and adjusted to permit the tracked `data/README.md`.

### Fixed — production-grade pass

- 5 bare `except:` clauses → `except Exception:` in
  `analyze_sky_exposure_streets.py`, `compare_areas.py`,
  `compute_deprivation_index_raster.py`, `compute_occupancy_density.py`,
  `compute_sectional_porosity.py`.
- `src/cfd_integration/io.py:96` — `raise ImportError` inside an
  `except ImportError` block now uses `from exc` so the original
  traceback is preserved.

### Known issues — flagged for follow-up

- 3 scripts import deleted sibling modules and fail at launch
  (`compute_solar_access_streets.py`, `analyze_sky_exposure_streets.py`,
  `compute_deprivation_streets.py`). Library functions still
  exist; documented in `scripts/README.md` for a future cleanup
  pass.
- `compute_deprivation_index.py` ↔ `compute_deprivation_index_raster.py`
  functional overlap; should likely consolidate on a
  `--resolution {unit,raster}` flag.

### Added — wind-rose work (earlier this cycle)

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
