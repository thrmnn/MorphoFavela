# Local setup — concentrated troubleshooting

Most of the install is in
[`README.md`](../../README.md#installation) and
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#development-setup). This
document covers the friction points: GDAL, conda vs venv, GPU stack,
and the common errors that show up on a fresh machine.

## The 30-second path (when everything works)

```bash
git clone https://github.com/thrmnn/MorphoFavela.git && cd MorphoFavela
conda create -n morphofavela python=3.11 && conda activate morphofavela
pip install -e ".[dev]"
pytest tests/ -m "not integration" -q --tb=short
# → 508 tests pass; 69 integration tests deselected
```

If this works, you're done. If `pip install` or `pytest` fails, read on.

## GDAL / GEOS — the most common blocker

`geopandas`, `rasterio`, `pyogrio`, `fiona` all bind to native GDAL +
GEOS libraries. The Python wheel installs the bindings; the system
libraries must already be present.

| Platform | Install |
|---|---|
| Linux (Debian / Ubuntu) | `sudo apt install libgdal-dev gdal-bin libgeos-dev` |
| Linux (Fedora / RHEL) | `sudo dnf install gdal-devel geos-devel` |
| macOS (Homebrew) | `brew install gdal geos` |
| Windows | Use conda — `conda install -c conda-forge gdal geos`. WSL is the alternative. |

After installing the system libraries, **re-run** `pip install -e ".[dev]"` —
the wheel build needs to find them.

### "ImportError: libgdal.so.32: cannot open shared object file"

Your system GDAL is older than the wheel. Two options:

1. **Recommended:** use conda — `conda install -c conda-forge geopandas rasterio` will pin to versions matching the conda GDAL build.
2. **Alternative:** match Python wheel versions to your system GDAL: `gdalinfo --version` reports yours, then `pip install "geopandas<X" "rasterio<Y"`.

### GEOS version mismatch on macOS Sonoma

Homebrew GEOS 3.12 conflicts with shapely's bundled GEOS in some pip
wheels. Fix:

```bash
pip uninstall shapely -y
pip install --no-binary shapely shapely
```

This forces shapely to compile against your system GEOS.

## Conda vs venv

Use conda when:

- You're on Windows (GDAL via conda is the path of least resistance).
- You want one command to install GDAL + GEOS + Python + bindings.
- You need PyTorch3D for the GPU SVF (the conda channel ships pre-built CUDA wheels).

Use venv when:

- Linux and you already have `libgdal-dev` from apt.
- You want a smaller environment.
- You're producing a wheel for distribution.

## GPU stack (optional — for `src/svf_v2` ray-cast)

```bash
pip install -e ".[gpu]"        # adds torch + pytorch3d
```

Most users don't need this. The CPU SVF path (joblib parallel raycast
in `src/svf_v2/compute.py`) is the production path; the GPU path is
optional and was validated against the CPU path in
[`docs/GPU_SVF_EXACT_VALIDATION.md`](../GPU_SVF_EXACT_VALIDATION.md).

If you do need GPU and `pip install` of `pytorch3d` fails:

```bash
# Install PyTorch first against your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install PyTorch3D from source (the pip wheel often lags PyTorch)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## "ModuleNotFoundError: No module named 'src'"

You forgot the `-e` (editable). Re-install:

```bash
pip install -e ".[dev]"
```

Confirm: `python -c "import src.morphometry; print(src.morphometry.__file__)"`
should print a path inside this repo.

## "ModuleNotFoundError: No module named 'umep_processing'"

Only relevant if you're running `scripts/validate_svf_against_umep.py`.
The validator vendors UMEP-processing on first run into
`vendor/umep_processing/` (gitignored). If the auto-vendoring fails:

```bash
git clone https://github.com/UMEP-dev/UMEP-processing.git vendor/umep_processing
# Then re-run the validator — it patches the QGIS-only __init__.py automatically
```

## "pyogrio.errors.DataSourceError: ... CRS not recognised"

Your GDAL is missing PROJ.6 + EPSG database. Fix:

```bash
# Linux
sudo apt install proj-bin proj-data

# macOS
brew install proj
```

Then `pip install --force-reinstall pyogrio`.

## Smoke-test failures

```bash
pytest tests/ -m "not integration" -q --tb=short
```

If this fails, the **first** failure usually points at the cause. The
most common causes (in order):

1. **GDAL/GEOS binding** — see "ImportError" sections above.
2. **`numpy < 2.3` enforcement** — UMEP validator vendoring requires `numba`-compatible numpy. If you've installed `numpy 2.3+` and pytest collection errors out: `pip install "numpy<2.3"`.
3. **`pytest-mark` not finding `fast` / `integration` markers** — you're using a stale pytest cache. `rm -rf .pytest_cache` and re-run.
4. **File-modification race in tests writing under `outputs/`** — the test `tests/test_cfd_integration/test_analyze.py` uses tmp dirs; if you see a "file already exists" error, you've got a stale `outputs/_test_*` from a prior interrupted run. `rm -rf outputs/_test_*` and re-run.

## Data — what you need before running the pipeline

Per-site inputs live under `data/{site}/` (gitignored). The contract
is in [`data/README.md`](../../data/README.md). For a fresh clone,
`data/` is empty — you need to obtain (or be given) the per-site
shapefiles + DTM rasters before running the pipeline. The five
campaign sites are `vidigal`, `rocinha`, `riodaspedras`,
`complexo_do_alemao`, and `maré`.

You **can** still run the smoke-test suite without any `data/` —
the tests use synthetic geometry under `tests/`.

## Wind input — INMET BDMEP quirks

If you're rebuilding `wind_rose.json` from raw INMET archives, three
quirks bite:

1. **The INMET server cuts large transfers from one IP.** Download
   yearly ZIPs separately, not in one batch. `scripts/download_inmet_zips.py`
   handles this.
2. **Date format changed mid-archive** — pre-2019 uses `YYYY-MM-DD`,
   2019+ uses `YYYY/MM/DD`. The extractor handles both, but a custom
   ingestion will need to too.
3. **Column names are accent-bearing** — `direção_horaria`, not
   `direcao_horaria`. Read with `encoding='latin-1'`.

`scripts/build_wind_rose.py` (via `extract_inmet_stations.py`) already
handles all three quirks; reuse it rather than re-implementing.

## When in doubt

- File a GitHub issue with: `python -V`, OS + version, conda vs venv, the full traceback, and what you tried.
- Check `data/{site}/PROVENANCE.md` (when present) — many "bugs" are upstream raster gaps documented there.
