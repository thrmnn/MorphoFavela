# Next Steps: GPU SVF Acceleration Testing

## ✅ Current Status

**Dependencies Installed:**
- ✓ PyTorch 2.5.1+cu121 (CUDA 12.1)
- ✓ PyTorch3D 0.7.9
- ✓ GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.6 GB)
- ✓ CUDA available and working

**Implementation:**
- ✓ GPU computation module complete
- ✓ GPU-enabled script ready
- ✓ Comparison tools ready
- ✓ Test script prepared

## 🎯 Next Steps

### Step 1: Run GPU-Accelerated SVF Computation for riodaspedras

**Option A: Use the automated test script**
```bash
cd /home/theo/IVF
source /home/theo/miniconda3/etc/profile.d/conda.sh
conda activate IVF
./scripts/test_gpu_svf_riodaspedras.sh
```

**Option B: Run manually**
```bash
cd /home/theo/IVF
source /home/theo/miniconda3/etc/profile.d/conda.sh
conda activate IVF

python scripts/compute_svf_streets_gpu.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --roads data/riodaspedras/raw/roads_riodaspedras.shp \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --area riodaspedras \
    --use-gpu \
    --spacing 3.0 \
    --height 1.5 \
    --sky-patches 145 \
    --gpu-batch-size 100 \
    --gpu-samples-per-ray 50
```

**Expected Output:**
- `outputs/riodaspedras/svf_streets_gpu/street_svf_points.gpkg`
- `outputs/riodaspedras/svf_streets_gpu/street_svf_segments.gpkg`
- `outputs/riodaspedras/svf_streets_gpu/street_svf_statistics.csv`
- `outputs/riodaspedras/svf_streets_gpu/street_svf_map.png`
- `outputs/riodaspedras/svf_streets_gpu/street_svf_distribution.png`

### Step 2: Compare GPU Results with Previous CPU Results

After GPU computation completes, compare with existing CPU results:

```bash
python scripts/compare_svf_cpu_gpu.py \
    --cpu-results outputs/riodaspedras/svf_streets/street_svf_points.gpkg \
    --gpu-results outputs/riodaspedras/svf_streets_gpu/street_svf_points.gpkg \
    --output-dir outputs/riodaspedras/svf_streets_comparison
```

**This will generate:**
- Comparison statistics (mean, std, correlation)
- Scatter plot (CPU vs GPU SVF values)
- Difference histogram
- Comparison CSV with detailed differences

### Step 3: Validate Results

**Check the following:**

1. **Correlation**: Should be >0.95 (high correlation between CPU and GPU)
2. **Mean Absolute Difference**: Should be <0.05 (acceptable accuracy)
3. **Computation Time**: GPU should be significantly faster
4. **Visual Inspection**: Review comparison plots

**Expected Performance:**
- **CPU**: ~7 seconds per point → ~65 hours for 33,387 points
- **GPU (RTX 4060)**: ~0.1-0.5 seconds per point → ~1-3 hours for 33,387 points
- **Speedup**: 10-70× depending on batch size and sampling

### Step 4: Optimize Parameters (if needed)

Based on test results, you may need to adjust:

- **`--gpu-batch-size`**: Increase if GPU memory allows (default: 100)
- **`--gpu-samples-per-ray`**: Increase for accuracy, decrease for speed (default: 50)
- **`--sky-patches`**: Same as CPU version (145 for standard, 300 for high precision)

### Step 5: Document Results

After successful testing:
1. Document performance improvements
2. Note any accuracy differences
3. Update implementation status
4. Commit test results (if desired)

## 📊 What to Monitor

During GPU computation, monitor:
- GPU memory usage: `watch -n 1 nvidia-smi`
- Computation progress (shown in progress bar)
- Any warnings or errors
- Final computation time

## 🔧 Troubleshooting

**If GPU computation fails:**
- Check GPU memory: Reduce `--gpu-batch-size`
- Check accuracy: Increase `--gpu-samples-per-ray`
- Fallback: Remove `--use-gpu` flag to use CPU

**If results differ significantly:**
- Increase `--gpu-samples-per-ray` for better accuracy
- Check threshold in `_check_points_visible()` function
- Compare with CPU results point-by-point

## 📝 Quick Reference

**Data Files:**
- STL: `data/riodaspedras/raw/full_scan.stl`
- Roads: `data/riodaspedras/raw/roads_riodaspedras.shp`
- Buildings: `data/riodaspedras/raw/riodaspedras_buildings.shp`

**Previous CPU Results:**
- Points: `outputs/riodaspedras/svf_streets/street_svf_points.gpkg`
- Segments: `outputs/riodaspedras/svf_streets/street_svf_segments.gpkg`

**New GPU Results:**
- Points: `outputs/riodaspedras/svf_streets_gpu/street_svf_points.gpkg`
- Segments: `outputs/riodaspedras/svf_streets_gpu/street_svf_segments.gpkg`

## 🚀 Ready to Test!

All dependencies are installed and verified. You can now proceed with Step 1.
