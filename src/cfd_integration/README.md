# CFD Integration Module — Result Format Specification

This document is the **contract** between the OpenFOAM CFD simulation repo and
this analysis repo. The CFD agent must produce outputs matching this schema
so that `src/cfd_integration/` can load, aggregate, and analyse them without
modification.

---

## Required outputs per simulation

Each simulation = **one patch × one wind direction**. For each, produce:

### Directory layout

```
data/{site}/cfd_results/{patch_id}/{wind_direction}/
  sample_points.csv       # REQUIRED — primary data
  summary.json            # REQUIRED — simulation metadata
  field.vtu               # OPTIONAL — full 3D field for deep dives
```

- `site`: one of `vidigal`, `rocinha`, `riodaspedras`, `complexo_do_alemao`, `maré`
- `patch_id`: from `outputs/{site}/sampling_cfd/campaign_sampling/campaign_patches.csv` (e.g., `VDG-P01`, `MAR-P07`)
- `wind_direction`: one of `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW` (meteorological bearing from north)

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

**Placeholder files already exist at this location** generated from a
climatological prior for Rio de Janeiro. They MUST be replaced with real
INMET station data before running the CFD campaign — the source field in
the existing files is explicitly tagged `PLACEHOLDER` so the swap is
obvious.

To rebuild a wind rose from INMET CSV observations:

```bash
python scripts/build_wind_rose.py --site vidigal \
    --inmet-csv data/inmet/alto_da_boa_vista.csv \
    --station "INMET Alto da Boa Vista A652" \
    --year-start 2010 --year-end 2023
```

Schema (matches `src/cfd_integration/schema.py:WindRose`):

```json
{
    "site": "vidigal",
    "source": "INMET Alto da Boa Vista A652 2010-2023 — n=113,892 obs",
    "frequencies": {"N": 0.04, "NE": 0.12, ..., "NW": 0.09},
    "mean_speeds":  {"N": 2.1,  "NE": 3.4,  ..., "NW": 1.8},
    "recommended_station": "Alto da Boa Vista (A652) — nearest mountainous site"
}
```

Frequencies sum to ~1.0. `mean_speeds` in m/s at 10 m height. Used to
compute annualised (wind-rose-weighted) metrics per patch.

---

## Methodology requirements for the CFD agent

These are the decisions that affect scientific validity. The analysis code
assumes all of the following:

### Simulation domain

- **Circular domain, radius 250 m** around each patch center (per the sampling script). Cylindrical OpenFOAM domain — reuse mesh across wind directions by rotating inlet BC.
- **Pedestrian sampling is only valid within the central 100 m analysis
  patch** (50 m from center in each direction). The outer 150 m of the domain
  is for atmospheric flow development and should not be used for quantitative
  results. This is enforced in `src/cfd_integration/aggregate.py`.

### Turbulence and boundary conditions

- **Turbulence model**: k-ω SST (standard for atmospheric boundary layer flows
  at this scale). RANS, not LES.
- **Inlet**: logarithmic velocity profile scaled to `wind_speed_ref` at 10 m
  height, with turbulence intensity matching suburban ABL (I ≈ 0.15 at 10 m).
- **Outlet**: `zeroGradient` for U, k, ω; `fixedValue` p=0.
- **Top**: `slip` or `symmetryPlane` at domain height (at least 5 × H_max, i.e.,
  ≥ 75 m for these sites).
- **Ground + buildings**: no-slip walls. Use `nutkRoughWallFunction` on the
  ground with roughness z₀ ≈ 0.03 m (suburban surface class).
- **Domain height**: minimum 6 × H_max to prevent blockage effects (≥ 90 m for
  these sites).
- **Blockage ratio**: each patch's `H_max_analysis` is in its `patch_meta.json`.
  The 250 m radius × 90 m height domain gives blockage << 3% for all patches.

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
