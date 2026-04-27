# Favela Morphometric Analysis

A Python pipeline for calculating morphometric metrics from building footprints with height attributes. Designed for analyzing informal settlement geometry and urban morphology.

## Features

- **Multi-area analysis**: Support for comparing multiple study areas (formal vs informal settlements)
- **Comprehensive metrics**: Building morphometry, extended morphology indicators, environmental performance (SVF, solar access, porosity), and deprivation indices
- **Area-based filtering**: Automatic filtering policy (applied to informal, skipped for formal areas)
- **Comparative analysis**: Automated comparison framework with statistical tests and PDF reports
- **Rich visualizations**: Thematic maps, statistical distributions, side-by-side comparisons
- **Professional reports**: Clean, academic-style PDF reports with comprehensive findings
- **Robust validation**: Validates data quality, CRS, and geometry before processing

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed project roadmap. **Current status**: Phases 1-4 complete. Phase 5 (CFD campaign) — 119 patches allocated across 5 sites, ready for OpenFOAM simulations. Paper figures for Nature Cities submission complete.

## Repository Map

This repo handles morphometric analysis, sampling, wind input data, and paper
figures. CFD simulations themselves run in a separate repo.

| Concern | Location |
|---|---|
| **Morphometric analysis** | `src/`, `scripts/calculate_*.py`, `scripts/run_svf_v2.py`, … |
| **CFD patch sampling** (5 sites × 119 patches) | `scripts/run_pilot_sampling.py`, `scripts/run_campaign_sampling.py` |
| **CFD I/O contract** (what CFD must produce, how we ingest) | `src/cfd_integration/README.md` |
| **CFD simulation execution** | **Separate repo at `~/Airflow`** (OpenFOAM + SLURM on MIT ORCD) |
| **Wind input** (boundary conditions, annual weighting) | `scripts/download_inmet_zips.py` → `scripts/extract_inmet_stations.py` → `scripts/build_wind_rose.py` → `data/{site}/wind_rose.json` |
| **Technical report** (canonical deliverable) | `docs/technical_report/technical_report.md` (and `.pdf`) |
| **Paper figures** (Nature Cities) | `outputs/paper_figures/` |

**End-to-end pipeline order** (per site):

1. `scripts/build_extended_context.py --area {site} --buffer 300` — clip city-wide buildings + DTM to a 300 m buffer.
2. Per-feature analyses (independent, can run in parallel):
   `run_svf_v2.py`, `compute_solar_access.py`, `compute_sectional_porosity.py`,
   `compute_occupancy_density.py`, `compute_urban_morphology.py`.
3. `scripts/run_pilot_sampling.py` → `scripts/run_campaign_sampling.py` — stratify on SVF × slope × λp, allocate 22-25 patches per site.
4. **CFD execution** lives in `~/Airflow` (`scripts/hpc/submit_patch_chain.sh PATCH_ID`). Returns `data/{site}/cfd_results/{patch_id}/{wind_dir}/`.
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
conda create -n IVF python=3.11 && conda activate IVF
pip install -e ".[dev]"
```

GDAL / GEOS native libraries must be present (`apt install libgdal-dev`
on Linux, `brew install gdal` on macOS) before `pip install`.
For GPU SVF: `pip install -e ".[gpu]"`.

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

CFD runs live in `~/Airflow`. Drop the patch artefacts in there and
submit:

```bash
# ~/Airflow:
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

## Installation

```bash
git clone https://github.com/thrmnn/MorphoFavela.git && cd MorphoFavela
conda create -n IVF python=3.11 && conda activate IVF
pip install -e ".[dev]"          # add ".[gpu]" for PyTorch3D-backed SVF
```

GDAL / GEOS native libraries are required (`apt install libgdal-dev`
on Linux, `brew install gdal` on macOS). See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for pre-commit setup and the
test/lint workflow.

The full input contract — what each `data/{site}/` subdirectory must
contain and where each file comes from — is documented in
[`data/README.md`](data/README.md).

## Project Structure

```
IVF/
├── README.md, ROADMAP.md, CLAUDE.md      # Entry points (see also Repository Map above)
│
├── src/                                  # Library code (importable)
│   ├── morphometry/                      # Core morphometric metrics
│   ├── svf_v2/                           # GPU sky-view-factor computation
│   ├── solar/                            # Façade + ground solar access
│   ├── cfd_integration/                  # CFD I/O contract, wind weighting (see its README)
│   ├── visualization/                    # Map + chart helpers
│   ├── urban_morphology.py               # BCR, FAR, λp, λf
│   ├── typology.py                       # Settlement typology
│   ├── exposure.py                       # Sky-exposure-plane exceedance
│   ├── spatial_analysis.py               # Moran's I, LISA, Gi*
│   └── config.py                         # Filtering thresholds + plot settings
│
├── scripts/                              # Executable entry points
│   ├── build_extended_context.py         # 300 m buffer per site (run first)
│   ├── run_svf_v2.py                     # SVF (GPU)
│   ├── compute_solar_access.py           # Ground solar
│   ├── run_facade_solar.py               # Façade solar
│   ├── compute_sectional_porosity.py     # Porosity
│   ├── compute_occupancy_density.py      # Density proxy
│   ├── compute_urban_morphology.py       # Zone-level metrics
│   ├── analyze_sky_exposure_streets.py   # Sky exposure plane
│   ├── compute_deprivation_index*.py     # Combined indices
│   ├── compare_areas.py                  # Formal vs informal report
│   ├── run_pilot_sampling.py             # 12-stratum CFD pilot (12-15 patches)
│   ├── run_campaign_sampling.py          # Full CFD campaign top-up (22-25 patches)
│   ├── build_wind_rose.py                # INMET / Iowa ASOS → wind_rose.json
│   ├── extract_inmet_stations.py         # Pull station CSVs from yearly INMET ZIPs
│   ├── hpc/                              # SLURM helpers (most CFD HPC code is in ~/Airflow)
│   └── data_utils/, debug/               # Small helpers
│
├── data/        # gitignored — site rasters, footprints, INMET ZIPs, wind roses
├── outputs/     # gitignored — analysis artefacts (paper_figures/*.py is tracked)
└── docs/
    ├── technical_report/                 # Canonical deliverable (md + pdf)
    ├── guides/                           # Per-feature usage guides
    └── archive/                          # Superseded planning + summary docs
```

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
- **Pinned dependencies**: see [`pyproject.toml`](pyproject.toml).
- **Linting + formatting**: ruff is gating CI; configuration is in
  the `[tool.ruff]` section of `pyproject.toml`.

## Documentation

- **Technical report** (canonical project description, distributed
  with the code): [`docs/technical_report/technical_report.md`](docs/technical_report/technical_report.md)
  + [`.pdf`](docs/technical_report/technical_report.pdf). Rebuild
  with `python docs/technical_report/build_pdf.py`.
- **Roadmap + project status**: [`ROADMAP.md`](ROADMAP.md).
- **Changelog**: [`CHANGELOG.md`](CHANGELOG.md).
- **Per-feature usage guides**: [`docs/guides/`](docs/guides/).
- **Superseded planning docs** (kept for archaeology, not a
  current reference): [`docs/archive/`](docs/archive/).

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
