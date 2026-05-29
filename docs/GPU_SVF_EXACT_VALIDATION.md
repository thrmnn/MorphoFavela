# GPU SVF Exact Intersection Validation

## What Changed

The GPU SVF kernel was upgraded from sampling-based obstruction checks to exact
ray-triangle intersection (Moller-Trumbore) with chunked execution on GPU.

This aligns GPU semantics with the CPU reference (`mesh.ray_trace`-based)
instead of using approximate point-to-mesh proximity tests.

## Why

The previous approximation could diverge from CPU ground truth in dense urban
geometry, especially around narrow streets and complex building edges.

The exact kernel fixes those mismatches while still using GPU acceleration.

## Validation Results

Ground-truth CPU references:

- `outputs/riodaspedras/svf_streets/street_svf_points.gpkg`
- `outputs/vidigal_tls/svf_streets/street_svf_points.gpkg`

Validated exact-GPU outputs:

- `outputs/riodaspedras/svf_streets_gpu_exact/street_svf_points.gpkg`
- `outputs/vidigal_tls/svf_streets_gpu_exact/street_svf_points.gpkg`

Comparison summary:

- Rio das Pedras: correlation 1.0000, max abs diff 0.0065
- Vidigal_TLS: correlation 1.0000, max abs diff 0.0000

## Final Production Run

After validation on both reference areas, the exact GPU pipeline was run for
Rocinha:

- `outputs/rocinha/svf_streets_gpu_exact/street_svf_points.gpkg`
- `outputs/rocinha/svf_streets_gpu_exact/street_svf_segments.gpkg`
- `outputs/rocinha/svf_streets_gpu_exact/street_svf_statistics.csv`

## Repro Commands

Use the same conda environment used by project scripts.

### Rio das Pedras (Exact GPU)

```bash
python scripts/compute_svf_streets_gpu.py \
  --stl data/riodaspedras/raw/full_scan.stl \
  --roads data/riodaspedras/raw/roads_riodaspedras.shp \
  --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
  --area riodaspedras \
  --spacing 3.0 \
  --height 1.5 \
  --sky-patches 145 \
  --use-gpu \
  --gpu-batch-size 200 \
  --gpu-ray-chunk-size 1024 \
  --gpu-tri-chunk-size 4096 \
  --output-dir outputs/riodaspedras/svf_streets_gpu_exact
```

### Vidigal_TLS (Exact GPU)

```bash
python scripts/compute_svf_streets_gpu.py \
  --stl data/vidigal_tls/raw/full_scan.stl \
  --roads data/vidigal_tls/raw/roads_vidigal.shp \
  --footprints data/vidigal_tls/raw/vidigal_buildings.shp \
  --area vidigal_tls \
  --spacing 3.0 \
  --height 1.5 \
  --sky-patches 145 \
  --use-gpu \
  --gpu-batch-size 200 \
  --gpu-ray-chunk-size 1024 \
  --gpu-tri-chunk-size 4096 \
  --output-dir outputs/vidigal_tls/svf_streets_gpu_exact
```

### Rocinha (Exact GPU)

```bash
python scripts/compute_svf_streets_gpu.py \
  --stl data/rocinha/rocinha.stl \
  --roads data/rocinha/raw/roads_rocinha.shp \
  --footprints data/rocinha/raw/rocinha_buildings.shp \
  --spacing 3.0 \
  --height 1.5 \
  --sky-patches 145 \
  --use-gpu \
  --gpu-batch-size 200 \
  --gpu-ray-chunk-size 1024 \
  --gpu-tri-chunk-size 4096 \
  --output-dir outputs/rocinha/svf_streets_gpu_exact
```

