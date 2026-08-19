# MorphoFavela — Morphometric analysis of informal settlements

A Python pipeline for the morphometric and CFD-coupled analysis of pedestrian-level wind, solar, and ventilation in dense informal urban form. Targeted at five Rio de Janeiro favelas (Vidigal, Rocinha, Complexo do Alemão, Rio das Pedras, Maré) but written to onboard new sites without code changes.

**What this repo does**

- Builds a **10 m morphometric grid** (12 indicators per cell — SVF, λp, λf, σh, slope, aspect, porosity, … — materialised as 25 columns since λf is stored per-direction) from building footprints + DTM
- Generates a **stratified CFD sampling campaign** (119 patches across 5 sites, 12-strata SVF × slope × λp grid, 80 m maximin spacing, 250 m circular domain)
- Ingests measured **wind input** (INMET BDMEP + Iowa ASOS METAR) into per-site `wind_rose.json` for annual weighting
- Specifies a **CFD I/O contract** and ingests OpenFOAM-derived wind fields when they return from the simulation cluster
- Produces the **technical report** ([`docs/technical_report/`](docs/technical_report/)) and Nature Cities paper figures

**What this repo does NOT do**

- Run OpenFOAM. Mesh generation, case setup, and HPC submission live in a separate repo at `~/SCL/SCR/Airflow` (MIT ORCD).
- Host the manuscript. The Nature Cities draft lives elsewhere; this repo provides the technical report that backs it.

**Audience.** Engineers, researchers, and reviewers who need to read the methodology, reproduce a figure, validate a number, or onboard a new site. Start with [`docs/technical_report/technical_report.md`](docs/technical_report/technical_report.md) for the methodology; this README is the operational guide.

## Status

See [ROADMAP.md](ROADMAP.md) for the full roadmap and version history. **As of June 2026**: 5 campaign sites + 3 calibration sites (Borel, Jacarezinho, Morro do Juramento) onboarded; 119-patch CFD campaign sampled and exported; wind input complete; result-side analysis pipeline shipped + synthetic-validated end-to-end on all 5 sites; SVF cross-validated against UMEP `svfForProcessing153` (closes the §10.3 limitation). CFD pilot ladder staged in `~/SCL/SCR/Airflow`; ingestion layer plumbed and waiting on first real return.

**Dissolved λf is now canonical.** Party-wall-corrected frontal-area density (touching footprints unioned into blocks before projection) is the single authoritative source, pinned bit-for-bit in `outputs/brisa_ventilation_fix/lambda_f_canonical.json` (gitignored output; regenerable) and test-locked (`tests/test_lambda_f_lockfile.py`). Pooled built mask n = 64,389; Oke / Grimmond-Oke flow-regime split ≈ 65 % skimming / 30 % wake / 5 % isolated. On top of it: **k = 6 cell morphotypes + k = 3 block morphotopes**; a typology→WHO-2 h-sun-failure predictor where the re-baseline **flipped** the headline — the continuous fabric vector (LOSO AUC-PR ≈ 0.84) outperforms the discrete type (≈ 0.61), with spatially-blocked block-bootstrap CIs (which overlap, so the flip is a direction not a clean gap) and low VIF; the blind risk map is externally validated on 3 calibration favelas (pooled AUC-PR 0.76). New geometric scalars: lateral-connectivity (distance-to-open-edge), 2-D ventilation-susceptibility (regime × depth), and effective wind-exposure (directional λf × wind rose). Roughness z0/zd via UMEP is **out of validity envelope for 53–75 % of cells** — that envelope is the result, not a defect. Track C (terrain-following morphometry) resolved as a scoping null. See [`docs/council_roadmap_2026-06-25.md`](docs/council_roadmap_2026-06-25.md), [`docs/repo_parallel_plan_2026-06-26.md`](docs/repo_parallel_plan_2026-06-26.md), and [`docs/brisa_round4_results_2026-06-25.md`](docs/brisa_round4_results_2026-06-25.md) for the standing plan and round-4/5 results.

## Repository Map

This repo handles morphometric analysis, sampling, wind input data, and paper
figures. CFD simulations themselves run in a separate repo.

| Concern | Location |
|---|---|
| **Morphometric analysis** | `src/morphometry/`, `src/svf_v2/`, `scripts/run_morphometric_audit.py`, `scripts/run_svf_v2.py`, … |
| **CFD patch sampling** (5 sites × 119 patches) | `scripts/run_pilot_sampling.py`, `scripts/run_campaign_sampling.py` |
| **CFD I/O contract** (what CFD must produce, how we ingest) | `src/cfd_integration/README.md` |
| **CFD simulation execution** | **Separate repo at `~/SCL/SCR/Airflow`** (OpenFOAM + SLURM on MIT ORCD) |
| **Wind input** (boundary conditions, annual weighting) | `scripts/download_inmet_zips.py` → `scripts/extract_inmet_stations.py` → `scripts/build_wind_rose.py` → `data/{site}/wind_rose.json` |
| **Technical report** (canonical deliverable) | `docs/technical_report/technical_report.md` (and `.pdf`) |
| **Paper figures** (Nature Cities) | `outputs/paper_figures/` |

**End-to-end pipeline order** (per site):

1. `scripts/build_extended_context.py --area {site} --buffer 300` — clip city-wide buildings + DTM to a 300 m buffer.
2. Per-feature analyses (independent, can run in parallel):
   `run_svf_v2.py`, `compute_solar_access.py`, `compute_sectional_porosity.py`,
   `compute_occupancy_density.py`, `compute_urban_morphology.py`.
3. `scripts/run_pilot_sampling.py` → `scripts/run_campaign_sampling.py` — stratify on SVF × slope × λp, allocate 22-25 patches per site.
4. **CFD execution** lives in `~/SCL/SCR/Airflow` (`scripts/hpc/submit_patch_chain.sh PATCH_ID`). Returns `data/{site}/cfd_results/{patch_id}/{wind_dir}/`.
5. `src/cfd_integration/` ingests CFD outputs and weights them by the site's wind rose.

**Wind input flow** (one-time, per site):

```
INMET BDMEP yearly ZIPs (Brazilian network) ──┐
                                              ├─→ build_wind_rose.py ──→ data/{site}/wind_rose.json
Iowa ASOS METAR (Galeão SBGL for Maré) ───────┘                          (used for §5 weighting)
```

Stations: A652 Forte de Copacabana → vidigal/rocinha · A636 Jacarepaguá → riodaspedras · A621 Vila Militar → complexo_do_alemao · SBGL Galeão → maré.

## Pipeline walkthrough — vidigal end-to-end

This is the canonical "I just cloned the repo, run the pipeline on
one site" path. Substitute another site name for `vidigal` to repeat.

### 1. Environment

```bash
git clone https://github.com/thrmnn/MorphoFavela.git && cd MorphoFavela
conda create -n morphofavela python=3.11 && conda activate morphofavela
pip install -e ".[dev]"
```

GDAL / GEOS native libraries must be present (`apt install libgdal-dev`
on Linux, `brew install gdal` on macOS) before `pip install`.
A GPU SVF backend is **not currently available** (the v2 ray-caster is CPU-parallelised; a GPU port is future work). The `[gpu]` extra installs PyTorch3D but the backend raises `NotImplementedError`.

### 2. Inputs

Place the per-site shapefile + DTM under `data/vidigal/` per the
contract in [`data/README.md`](data/README.md). At minimum:

```
data/vidigal/raw/vidigal_buildings.shp   # footprints with height attrs
data/vidigal/dtm_extended_300m.tif       # manually clipped from RJ DTM
```

### 3. Wind input (one-off, shared by all sites)

```bash
python scripts/download_inmet_zips.py \
    --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 \
    --out-dir data/inmet/raw

python scripts/extract_inmet_stations.py \
    --zips-dir data/inmet/raw --out-dir data/inmet/processed \
    --stations A652 A636 A621 A602

python scripts/build_wind_rose.py --site vidigal \
    --inmet-csv data/inmet/processed/concat/A652_2015_2024.csv
```

Result: `data/vidigal/wind_rose.json` (used later for annual weighting
of CFD outputs).

### 4. Extended morphometric context

```bash
python scripts/build_extended_context.py --area vidigal --buffer 300
```

Result: `data/vidigal/buildings_extended_300m.gpkg`. The 300 m buffer
ensures CFD patches near the favela edge see the surrounding urban
fabric.

### 5. Per-feature analyses (independent — run in parallel if you like)

```bash
python scripts/run_svf_v2.py            --area vidigal --spacing 2.0
python scripts/compute_solar_access.py  --area vidigal --threshold 2.0
python scripts/compute_sectional_porosity.py --area vidigal --grid-spacing 2.0
python scripts/compute_urban_morphology.py   --area vidigal
python scripts/run_morphometric_audit.py     --area vidigal
```

Outputs land in `outputs/vidigal/` — see
[`src/svf_v2/README.md`](src/svf_v2/README.md),
[`src/solar/README.md`](src/solar/README.md), and
[`src/morphometry/README.md`](src/morphometry/README.md) for
per-module details on what each writes.

### 6. CFD patch sampling

```bash
python scripts/run_pilot_sampling.py --site vidigal \
    --buildings data/vidigal/buildings_extended_300m.gpkg \
    --dtm data/vidigal/dtm_extended_300m.tif

python scripts/run_campaign_sampling.py
```

Result: 22 patches under
`outputs/vidigal/sampling_cfd/campaign_sampling/patches/{PATCH_ID}/`,
each with `buildings.gpkg`, `terrain.tif`, and `patch_meta.json`.

### 7. CFD execution (separate repo)

CFD runs live in `~/SCL/SCR/Airflow`. Drop the patch artefacts in there and
submit:

```bash
# ~/SCL/SCR/Airflow:
scripts/hpc/submit_patch_chain.sh VDG-P07
```

Results return at `data/vidigal/cfd_results/{patch_id}/{wind_dir}/`
per the contract in
[`src/cfd_integration/README.md`](src/cfd_integration/README.md).

### 8. Annualised aggregation

Once CFD outputs land:

```python
from src.cfd_integration.io import load_campaign_results
from src.cfd_integration.weighting import weighted_by_wind_rose

campaign = load_campaign_results("vidigal")
annual = weighted_by_wind_rose(campaign, "data/vidigal/wind_rose.json")
```

### 9. Paper figures

```bash
for f in outputs/paper_figures/fig*.py; do python3 "$f"; done
```

Outputs land in `outputs/paper_figures/exports/` (PNG + SVG); the
PNGs already published in the technical report live in
`docs/technical_report/figures/`.

### 10. Site dashboards + distribution bundles

The pipeline outputs flow into shareable artefacts under
`outputs/_distribution/` (all gitignored):

```bash
# Canonical street-observer network — single source of truth for sampling
python scripts/build_street_observers.py --bundle      # writes observers + tarball

# Static A3 "Folha de Rua" per site (PNG + PDF + 8 atomic panels)
python scripts/build_site_dashboard.py --site vidigal

# Interactive HTML site sheet (Leaflet + Plotly + methodology drawer)
python scripts/build_html_dashboard.py --site vidigal

# 3D data bundle for an external solar-access collaborator
python scripts/build_mingze_bundle.py                  # 4 sites, ~37 MB tarball
```

The HTML dashboards surface the most recent audit (`docs/audit_2026-06-03*.md`)
findings inline — C1 length-weighting fix, H1-H4 sampling biases, M1-M6
schema and CLI fixes — so reviewers see them as toggles and callouts
rather than buried in an appendix. See
[`docs/workflow_patterns.md`](docs/workflow_patterns.md) for the
multi-agent audit-and-deliverable-per-entity pattern that produces them.

## Installation

The recommended path is Conda via the pinned [`environment.yml`](environment.yml),
which pulls the native GDAL / GEOS / PROJ stack that geopandas, rasterio,
and fiona link against:

```bash
git clone https://github.com/thrmnn/MorphoFavela.git && cd MorphoFavela
conda env create -f environment.yml      # python 3.11 + pinned deps + the package
conda activate morphofavela
```

Or a lighter unpinned setup (you supply GDAL / GEOS via `apt`/`brew`):

```bash
conda create -n morphofavela python=3.11 && conda activate morphofavela
pip install -e ".[dev]"          # ([gpu] extra exists but the GPU SVF backend is not yet ported — CPU is supported)
```

GDAL / GEOS native libraries are required for the pip path (`apt install
libgdal-dev` on Linux, `brew install gdal` on macOS). See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for pre-commit setup and the
test/lint workflow.

After `pip install -e .` the most-used scripts are available as
`mf-*` console commands (e.g. `mf-svf --area vidigal`,
`mf-wind-rose --site maré --asos-csv ...`). The walkthrough below
uses `python scripts/foo.py` form for clarity, but the two
invocations are interchangeable. See
[`scripts/README.md`](scripts/README.md) for the full mapping.

The full input contract — what each `data/{site}/` subdirectory must
contain and where each file comes from — is documented in
[`data/README.md`](data/README.md).

## Project Structure

```
MorphoFavela/
├── README.md, ROADMAP.md                 # Entry points (see also Repository Map above)
│
├── src/                                  # Library code (importable)
│   ├── morphometry/                      # Core metrics + morphotypes/signature, roughness z0/zd, prioritization
│   ├── svf_v2/                           # Sky-view-factor computation (CPU ray-casting)
│   ├── solar/                            # Façade + ground solar access
│   ├── cfd_integration/                  # CFD I/O contract, wind weighting (see its README)
│   ├── visualization/                    # Map + chart helpers
│   ├── viz/                              # Dashboard panel + map composition helpers
│   ├── urban_morphology.py               # BCR, FAR, λp, λf (dissolved / party-wall corrected; canonical)
│   ├── typology.py                       # Settlement typology
│   ├── exposure/                         # Sky-exposure-plane exceedance + deprivation
│   ├── spatial_analysis.py               # Moran's I, LISA, Gi*
│   └── config.py                         # Filtering thresholds + plot settings
│
├── scripts/                              # Executable entry points
│   ├── build_extended_context.py         # 300 m buffer per site (run first)
│   ├── run_svf_v2.py                     # SVF (CPU ray-casting)
│   ├── compute_solar_access.py           # Ground solar
│   ├── run_facade_solar.py               # Façade solar
│   ├── compute_sectional_porosity.py     # Porosity
│   ├── compute_occupancy_density.py      # Density proxy
│   ├── compute_urban_morphology.py       # Zone-level metrics
│   ├── compute_deprivation_index_raster.py # Combined deprivation raster
│   ├── compare_areas.py                  # Formal vs informal report
│   ├── run_pilot_sampling.py             # 12-stratum CFD pilot (12-15 patches)
│   ├── run_campaign_sampling.py          # Full CFD campaign top-up (22-25 patches)
│   ├── build_wind_rose.py                # INMET / Iowa ASOS → wind_rose.json
│   ├── extract_inmet_stations.py         # Pull station CSVs from yearly INMET ZIPs
│   ├── build_street_observers.py         # Canonical per-site observer networks
│   ├── build_site_dashboard.py           # A3 Folha de Rua site sheets
│   ├── build_html_dashboard.py           # Interactive HTML site dashboards
│   ├── build_mingze_bundle.py            # 3D-data bundle for collaborators
│   ├── compare_mingze_vidigal.py         # Cross-method solar validation (Ladybug)
│   ├── analyze_typology_predictor.py     # Typology → WHO-2h-sun predictor (continuous-vector flip)
│   ├── build_signature.py                # k=6 cell morphotypes + k=3 block morphotopes
│   ├── run_lateral_connectivity.py       # Distance-to-open-edge lateral ventilation scalar
│   ├── run_ventilation_susceptibility.py # 2-D regime × depth susceptibility map
│   ├── run_wind_exposure.py              # Effective wind-exposure (directional λf × wind rose)
│   ├── build_roughness.py                # Per-cell z0/zd via UMEP (validity-envelope flagged)
│   ├── brisa_ventilation/                # Numbered BRISA pipeline (λf repair → regime taxonomy → lock file)
│   ├── brisa_deck/                       # BRISA presentation figure scripts
│   ├── hpc/                              # SLURM helpers (most CFD HPC code is in ~/SCL/SCR/Airflow)
│   └── data_utils/, debug/               # Small helpers
│
├── data/        # gitignored — site rasters, footprints, INMET ZIPs, wind roses
├── outputs/     # gitignored — analysis artefacts (paper_figures/*.py is tracked)
└── docs/
    ├── technical_report/                 # Canonical deliverable (md + pdf + figures/)
    ├── methodology/                      # Standalone methodology docs (SVF, sky-exposure, indicators)
    ├── FAVELA_EXTRACTION_WORKFLOW.md     # GIS extraction workflow
    ├── GPU_SVF_EXACT_VALIDATION.md       # GPU-vs-CPU SVF parity report
    └── cfd_sampling_overrides.yaml       # Documented sampling-coverage gap downgrades
```

See [`docs/README.md`](docs/README.md) for a one-line summary of each.

## Configuration + per-module details

- **Filtering thresholds** for morphometric analysis (height /
  area / volume / h/w-ratio caps, percentile filters): `src/config.py`.
- **Script index** — every entry-point in `scripts/`, grouped by
  pipeline stage with its library backing:
  [`scripts/README.md`](scripts/README.md).
- **Per-feature internals** — methodology, output schema, public
  API:
  - [`src/svf_v2/README.md`](src/svf_v2/README.md) — Sky View Factor
  - [`src/solar/README.md`](src/solar/README.md) — Solar access (ground + façade)
  - [`src/morphometry/README.md`](src/morphometry/README.md) — 12-indicator 10 m grid + audit pipeline
  - [`src/cfd_integration/README.md`](src/cfd_integration/README.md) — CFD I/O contract + wind-rose weighting
  - [`src/visualization/README.md`](src/visualization/README.md) — building / zone-level chart helpers
- **Paper figures** (Nature Cities): see
  [`outputs/paper_figures/README.md`](outputs/paper_figures/README.md).
- **Dependencies**: declared in [`pyproject.toml`](pyproject.toml)
  (version floors); pinned exactly in [`environment.yml`](environment.yml)
  for reproduction.
- **Linting + formatting**: ruff is gating CI; configuration is in
  the `[tool.ruff]` section of `pyproject.toml`.

## Documentation

- **Technical report** (canonical project description, distributed
  with the code): [`docs/technical_report/technical_report.md`](docs/technical_report/technical_report.md)
  + [`.pdf`](docs/technical_report/technical_report.pdf). Rebuild
  with `python docs/technical_report/build_pdf.py`.
- **Methodology** (per-feature deep dives): [`docs/methodology/`](docs/methodology/)
  — sky-exposure plane, street-level SVF, the 12 morphometric indicators.
- **Workflow patterns**: [`docs/workflow_patterns.md`](docs/workflow_patterns.md)
  — the council-of-experts + judge-panel + per-entity build pattern
  used to produce the audit, site dashboards, and HTML deliverables.
- **Repo audits**: [`docs/audit_2026-06-03.md`](docs/audit_2026-06-03.md)
  + [`_round2.md`](docs/audit_2026-06-03_round2.md) — the two-pass
  audit that produced the recent length-weighted-SVF fix (commit
  d8175ca) and the dashboard caveat surfaces.
- **Doc map**: [`docs/README.md`](docs/README.md).
- **Roadmap + project status**: [`ROADMAP.md`](ROADMAP.md).
- **Changelog**: [`CHANGELOG.md`](CHANGELOG.md).

## Data availability

This repository ships the **analysis code, pipelines, paper-figure
scripts, and the technical report** — not the raw input geodata.

The building footprints, DTM rasters, and per-patch CFD inputs for the
five favelas are **not redistributed here**, for two reasons:

1. **Privacy.** Publishing precise footprints and coordinates of
   dwellings in informal settlements carries ethical risk for
   residents. We avoid releasing data that pinpoints individual
   structures.
2. **Licensing.** Some source rasters originate from municipal datasets
   whose terms do not permit redistribution.

**To reproduce results:**

- Supply your own inputs under `data/{site}/` following the contract in
  [`data/README.md`](data/README.md), or
- Contact the author (`thermann.ai@gmail.com`) for the processed inputs
  used in the paper. Derived, aggregated, non-identifying outputs are
  available on request, subject to a data-use agreement.

Wind input is built from public sources (INMET BDMEP and Iowa State
ASOS METAR) and is fully reproducible from the scripts in this repo.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development
environment setup, the test/lint workflow, commit conventions,
and the technical-report sync rule.

## License

[MIT](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff) — GitHub renders a "Cite this
repository" button from this file. The Nature Cities article DOI
will be added once published.
