# CFD Integration Module — Result Format Specification

This document is the **contract** between the OpenFOAM CFD simulation repo and
this analysis repo. The CFD agent must produce outputs matching this schema
so that `src/cfd_integration/` can load, aggregate, and analyse them without
modification.

---

## Required outputs per simulation

Each simulation = **one patch × one wind direction**. For each, produce one of two equivalent on-disk layouts; the IVF ingestion side auto-detects (`load_campaign_results`).

### Layout A — IVF native (CSV + cardinal direction dirs)

```
data/{site}/cfd_results/{patch_id}/{wind_direction}/
  sample_points.csv       # REQUIRED — primary data
  summary.json            # REQUIRED — simulation metadata
  field.vtu               # OPTIONAL — full 3D field for deep dives
```

- `wind_direction` ∈ `{N, NE, E, SE, S, SW, W, NW}` (meteorological bearing from north)

### Layout B — Airflow native (parquet + numeric-degree dirs)

```
data/{site}/cfd_results/{patch_id}/wind_{NNN}/
  *.parquet               # REQUIRED — primary data (one or more files; concatenated row-wise)
  summary.json            # REQUIRED — simulation metadata
```

- `wind_NNN` directory mapped to cardinal: `wind_000` → `N`, `wind_045` → `NE`, `wind_090` → `E`, `wind_135` → `SE`, `wind_180` → `S`, `wind_225` → `SW`, `wind_270` → `W`, `wind_315` → `NW`. Off-axis directories (anything other than the 8 listed) are skipped with a warning.
- Multiple parquet files in the same `wind_NNN/` directory are concatenated row-wise (used when the OpenFOAM agent emits one parquet per processor decomposition).
- `summary.json` `wind_direction` field must still use the cardinal form (`"NE"`, not `"045"`) — it describes the simulation, not the directory.

Common across both layouts:

- `site`: one of `vidigal`, `rocinha`, `riodaspedras`, `complexo_do_alemao`, `maré`
- `patch_id`: from `outputs/{site}/sampling_cfd/campaign_sampling/campaign_patches.csv` (e.g., `VDG-P01`, `MAR-P07`)
- Sample-row schema and `summary.json` schema are identical between the two layouts (Layout B's parquet uses the same column names as Layout A's CSV).

> **Synthetic results.** `scripts/generate_synthetic_cfd_results.py` emits this exact contract for pipeline exercise and figure prototyping before real OpenFOAM returns. Synthetic trees carry `"synthetic": true` plus a `provenance` block in every `summary.json` and are written to a **separate root** (`data/{site}/cfd_results_synthetic/`), never the ingestor-default `cfd_results/`. The loader ignores the extra keys, so a stray synthetic tree under `cfd_results/` would *not* self-flag — keep them physically separate.

### `sample_points.csv` (required)

A horizontal slice of the velocity/TKE field at **pedestrian height z = 1.5 m**
above ground, sampled on a regular grid at 2 m spacing covering the full 250 m
circular CFD domain.

**Required columns** (one row per sample point, ~49,000 rows per 250 m domain):

| Column | Unit | Description |
|--------|------|-------------|
| `x` | m | UTM easting (EPSG:31983) |
| `y` | m | UTM northing (EPSG:31983) |
| `z` | m | Sample elevation above ground (should be 1.5 for all rows) |
| `U` | m/s | Velocity x-component (east-positive) |
| `V` | m/s | Velocity y-component (north-positive) |
| `W` | m/s | Velocity z-component (up-positive) |
| `U_mag` | m/s | Velocity magnitude = √(U² + V² + W²) |
| `TKE` | m²/s² | Turbulent kinetic energy (from k-ω SST or RANS equivalent) |

**Optional column:**

| Column | Unit | Description |
|--------|------|-------------|
| `p` | Pa | Static pressure (useful for flow diagnostics) |

### `summary.json` (required)

Simulation metadata:

```json
{
    "patch_id": "VDG-P01",
    "site": "vidigal",
    "wind_direction": "NE",
    "wind_speed_ref": 5.0,
    "converged": true,
    "residual_final": 1.2e-5,
    "solver": "simpleFoam",
    "turbulence_model": "kOmegaSST",
    "n_iterations": 2000,
    "wall_clock_s": 4320.5
}
```

- `wind_speed_ref`: **inlet reference velocity** in m/s, measured at z = 10 m
  (standard meteorological reference height). All CFD velocities are
  dimensional; we scale per-direction later by the measured wind rose.
- `converged`: set `false` if residuals didn't meet target or the simulation
  was force-terminated. Non-converged simulations are flagged but not excluded.

### Wind rose (per site, one file)

```
data/{site}/wind_rose.json
```

**Placeholder files already exist at this location**, generated from a
Rio-coastal climatological prior. Each is explicitly tagged
`"quality_flag": "placeholder-prior"` and carries a per-site
recommended INMET station plus an `expected_adjustment` note; they MUST
be replaced with real station data before running the CFD campaign.

#### Getting INMET data

INMET publishes one nationwide yearly ZIP (~100 MB) per calendar year
at a stable URL. The ZIPs contain one CSV per automatic station for the
full year, BDMEP format (`sep=;`, `decimal=,`, latin-1, 8-row metadata
header, `-9999` for missing values, anemometer at z = 10 m).

```bash
# Download one year (requires browser User-Agent — default curl UA is blocked)
curl -A "Mozilla/5.0" -O \
  "https://portal.inmet.gov.br/uploads/dadoshistoricos/2023.zip"

# Unzip and keep only the 4 RJ stations we need
unzip -j 2023.zip 'INMET_SE_RJ_A6{52,36,21,02}*.CSV' -d data/inmet/2023/

# Build one rose from multiple yearly CSVs for the same station
python scripts/build_wind_rose.py --site vidigal \
    --inmet-csv data/inmet/2015_2024/A652_concat.csv \
    --station-id A652 --station-name "Forte de Copacabana" \
    --station-lat -22.988 --station-lon -43.190 \
    --year-start 2015 --year-end 2024
```

Recommended stations per site (verified April 2026):

| Site | Station | Code | Coords | Class |
|------|---------|------|--------|-------|
| Vidigal | Forte de Copacabana | A652 | −22.988, −43.190 | coastal |
| Rocinha | Forte de Copacabana | A652 | −22.988, −43.190 | coastal |
| Rio das Pedras | Jacarepaguá | A636 | −22.99, −43.37 | plain |
| Complexo do Alemão | Vila Militar | A621 | −22.86, −43.41 | urban interior |
| Maré | SBGL Galeão METAR (preferred) / A652 (INMET fallback) | — | −22.81, −43.25 | bayside |

Maré's bay regime is best captured by SBGL (Galeão airport) METAR via
Iowa State's ASOS archive; METAR ingestion is not yet implemented in
`build_wind_rose.py`. A621 and A602 coordinates were sourced from
cross-references rather than the INMET catalogue directly — verify
against the station catalogue CSV before citing.

#### JSON schema

```json
{
  "site": "vidigal",
  "source": "INMET Forte de Copacabana (A652) 2015-2024; n=84,321 obs (6,745 calm)",
  "frequencies": {"N": 0.04, "NE": 0.12, ..., "NW": 0.09},
  "mean_speeds":  {"N": 2.1, "NE": 3.4, ..., "NW": 1.8},
  "reference_height_m": 10.0,
  "station_id": "A652",
  "station_name": "Forte de Copacabana",
  "station_coords": [-22.988, -43.190],
  "time_window_start": "2015-01-01",
  "time_window_end": "2024-12-31",
  "n_observations": 84321,
  "calm_fraction": 0.08,
  "quality_flag": "measured"
}
```

Frequencies sum to 1.0. `mean_speeds` are m/s at z = 10 m after calm
exclusion. `calm_fraction` (observations with |U| < 0.5 m/s or NaN
direction) is recorded but NOT redistributed into the 8 directional
bins. `quality_flag` is one of `measured`, `gap-filled`, or
`placeholder-prior` — the last blocks accidental use for annualised
metrics.

**Neutral-stability assumption:** the k-ω SST inflow uses a log-law
profile that implicitly assumes a neutral atmospheric boundary layer.
For Rio, daytime unstable convection and evening stable inversions
bias the stagnation metric; this limitation is accepted for the
screening campaign. A future campaign could stratify the rose by
stability class.

---

## Methodology requirements for the CFD agent

These are the decisions that affect scientific validity. The analysis code
assumes all of the following:

### Simulation domain

**Rectangular per-direction (v1, May 2026).** Each patch × wind-direction
gets its own rectangular `blockMesh` sized per the rules in
[`rectangular_domain_v1.json`](rectangular_domain_v1.json) — a Franke /
COST 732 / Blocken (2015) wide-obstacle scheme. Headlines:

- Per-patch extents: `5·H_max + R` upstream, `15·H_max + R` downstream,
  `max(5·H_max + R, 5·W_patch) = 500 m` lateral each side, `5·H_max` top.
  R = 50 m (patch radius), W = 100 m (patch diameter).
- Lateral collapses to a uniform 500 m for all 119 patches because
  H_max < 90 m everywhere — so blockage is independent of H_max
  (uniformly 2 % silhouette envelope).
- Earlier cylindrical 250 m radius scheme is **deprecated** along with
  the `blocken_radius_required` indicator column.

**Pedestrian sampling is only valid within the central 100 m analysis
patch** (50 m from center in each direction). The rest of the domain is
for atmospheric flow development and is not consumed by `aggregate.py`.

### Turbulence and boundary conditions

- **Turbulence model**: k-ω SST (standard for atmospheric boundary layer flows
  at this scale). RANS, not LES.
- **Inlet**: logarithmic velocity profile scaled to `wind_speed_ref` at 10 m
  height, with turbulence intensity matching suburban ABL (I ≈ 0.15 at 10 m).
- **Outlet**: `zeroGradient` for U, k, ω; `fixedValue` p=0.
- **Top**: `slip` or `symmetryPlane` at domain height (= `5·H_max`, see
  manifest).
- **Ground + buildings**: no-slip walls. Use `nutkRoughWallFunction` on the
  ground with roughness z₀ ≈ 0.03 m (suburban surface class).
- **Blockage ratio**: gated at `< 0.05` (AIJ Tominaga 2008) on the silhouette
  envelope `D · H_max / (2 · lateral · top)`. With the v1 rectangular rule
  this is uniformly ≈ 0.02 across the campaign — see
  `domain_blockage_ratio` per row in `per_patch_indicators.csv`.

### Meshing

- Use **snappyHexMesh** with refinement zones around the analysis patch
  (cell size ≤ 1 m near buildings, coarsening outward).
- Building walls: 3 prism layers, y+ ≈ 30–300 for wall functions.
- Target total cell count: 3–10 M per patch. Patches vary — MAR-P07 is the
  simplest (1,200 buildings, flat), use it for initial mesh convergence study.

### Wind directions and weighting

- Simulate **all 8 cardinal directions per patch**: 8 × 119 patches = 952 sims.
- Each direction uses the same mesh; only the inlet rotation changes.
- Annualised metrics are computed on the analysis side using the local wind
  rose (frequency × speed weighting). The CFD agent does NOT need to weight
  directions — just deliver them separately.

### Convergence

- Target residuals: 1e-4 for U, k, ω; 1e-5 for p. Run to 2000–5000 iterations.
- Flag non-converged simulations in `summary.json` but still deliver the data.

### Post-processing

- Extract the horizontal slice at z = 1.5 m using `sample` utility in
  `system/sampleDict`:
  ```
  type sampledSet;
  axis xyz;
  setFormat raw;
  surfaceFormat vtk;
  fields (U k);
  sets
  (
      pedestrian_height
      {
          type uniform;
          axis xyz;
          spacing 2;
          start (x_min y_min 1.5);
          end   (x_max y_max 1.5);
      }
  );
  ```
  Convert the output to `sample_points.csv` with the columns above.

### Recommended test patch

**MAR-P07** (Maré, flat terrain, moderate density, 1,200 buildings). Validate
the full pipeline on this patch with 1 wind direction first. Once it converges
cleanly and the CSV output loads correctly in
`src/cfd_integration/io.py::load_patch_csv`, scale to all 8 directions, then
scale to the full 119-patch campaign.

---

## How the analysis repo ingests results

```python
from src.cfd_integration import load_campaign_results, aggregate_to_grid, weighted_by_wind_rose
from src.cfd_integration import load_grid, load_campaign_patches  # from fig_style helpers

# Load all simulations for a site
campaign = load_campaign_results(site="vidigal")

# Load the morphometric grid and patch locations
grid = load_grid("vidigal")
patches = load_campaign_patches("vidigal")

# For each patch, aggregate per-direction, then weight by wind rose
for patch_id in campaign.patch_ids():
    patch_row = patches[patches["patch_id"] == patch_id].iloc[0]
    center = (patch_row["center_x"], patch_row["center_y"])

    per_direction = {}
    for wind_dir in campaign.directions_for(patch_id):
        result = campaign.patches[patch_id][wind_dir]
        per_direction[wind_dir] = aggregate_to_patch(result, center)

    # Annualise
    annual = weighted_by_wind_rose(per_direction, campaign.wind_rose, weight_by="freq_speed")
    # annual now has keys like annual_U_mean, annual_stagnation_frac, annual_ach_patch
```

---

## Delivery checklist (for the CFD agent)

Before handing off results:

- [ ] All 119 patches × 8 directions = 952 simulations attempted
- [ ] Each `sample_points.csv` has the 8 required columns and z ≈ 1.5 for all rows
- [ ] Each `summary.json` includes `wind_speed_ref` and convergence flag
- [ ] Domain radius and mesh convergence verified on MAR-P07
- [ ] Wind rose JSON delivered for each site
- [ ] Loading test: `load_patch_csv("data/vidigal/cfd_results/VDG-P01/N/sample_points.csv")` succeeds
- [ ] Non-converged simulations flagged but included

---

## Contact

Format issues or methodology questions: open an issue tagged `cfd-integration`
in this repo.
