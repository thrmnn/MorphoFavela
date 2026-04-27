# `src/svf_v2/`

GPU-capable Sky View Factor (SVF) computation in world coordinates
(EPSG:31983). The "v2" replaces an earlier local-coordinate
implementation that conflated terrain and building origins; v2 keeps
everything in projected UTM throughout, which is required for slope-
sensitive ray-casting and for joining SVF outputs back to the rest of
the morphometric pipeline.

## What it does

For a given site (e.g., `vidigal`), the module:

1. Loads buildings + DTM and constructs a 3D scene as a triangle mesh
   in world coordinates.
2. Samples evaluation points (regular ground grid, street centreline,
   or building façade) per the analysis target.
3. For each point, casts rays into a Tregenza-discretised sky
   hemisphere and counts the fraction unobstructed.
4. Writes results as `.npy` arrays + GeoPackage-friendly CSVs.

CPU path uses pyvista's BVH; GPU path uses PyTorch3D mesh ray-mesh
intersections. The result is identical (validated to ~1e-3 SVF in the
exact-validation suite — see `docs/GPU_SVF_EXACT_VALIDATION.md`).

## Public API (re-exported in `__init__`)

| Function | Purpose |
|---|---|
| `compute_svf(...)` | Main entry: SVF for an array of points against a mesh |
| `compute_svf_raycasting(...)` | Lower-level ray-cast core; called by `compute_svf` |
| `generate_tregenza_patches(...)` | 145 / 290-patch sky discretisation |
| `sample_grid_points(...)` | Regular 2 D ground grid clipped to a polygon |
| `sample_street_points(...)` | Centreline samples along a road network |
| `sample_facade_points(...)` | Per-storey façade points (top of each floor) |
| `compute_facade_svf(...)` | SVF on building façade points |
| `compute_facade_solar_potential(...)` | Annual SVF-weighted solar exposure |
| `resolve_paths(area)` | Path resolver (DTM, footprints, output dir) |
| `save_grid_results(...)` | Writes SVF arrays + heatmap PNG + histogram |

## Submodules

- `compute.py` — ray-casting kernels (CPU and GPU)
- `scene.py` — mesh assembly from DTM + footprints
- `sampling.py` — grid / street / façade point generation
- `io.py` — output writers (NPY, CSV, PNG, STL)
- `paths.py` — area registry; **edit this file to add a new site**
- `utils.py` — shared helpers (mesh loading, ground masking)
- `facades.py` + `visualize.py` — façade analyses + plot helpers

## Typical usage

```bash
# Grid SVF for vidigal at 2 m spacing, all sky patches:
python scripts/run_svf_v2.py --area vidigal --mode grid --spacing 2.0
```

This calls `resolve_paths(area)` → `compute_svf(...)` → `save_grid_results(...)`.
For a custom run, import the functions directly:

```python
from src.svf_v2 import compute_svf, generate_tregenza_patches, resolve_paths

dtm, footprints, out_dir = resolve_paths("vidigal")
sky = generate_tregenza_patches(n=290)
svf = compute_svf(points, mesh, sky_directions=sky)
```

## Tests

`tests/test_svf_v2/` — 120+ tests covering Tregenza discretisation,
ray-casting against synthetic geometry, scene assembly, and visual
inspection (with golden PNG comparison).
