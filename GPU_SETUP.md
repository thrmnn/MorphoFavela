# GPU Acceleration Setup Guide

## Prerequisites

Before running GPU-accelerated SVF computation, you need to install PyTorch and PyTorch3D.

## Step 1: Install PyTorch with CUDA

### Check CUDA Version
```bash
nvcc --version
# or
nvidia-smi
```

### Install PyTorch

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU only (no GPU):
```bash
pip install torch torchvision torchaudio
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 2: Install PyTorch3D

### From Source (Recommended)
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### From PyPI (May have limited features)
```bash
pip install pytorch3d
```

### Verify Installation
```bash
python -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')"
```

## Step 3: Test GPU Setup

Run the GPU availability check:
```bash
python -c "from src.svf_gpu_utils import check_gpu_availability; print(check_gpu_availability())"
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--gpu-batch-size` (default: 100)
- Reduce `--gpu-samples-per-ray` (default: 50)
- Process fewer points at once

### PyTorch3D Installation Issues
- Ensure PyTorch version matches PyTorch3D requirements
- Try installing from source: `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`
- Check PyTorch3D GitHub for latest compatibility info

### No GPU Available
- The script will automatically fall back to CPU
- Use `--force-cpu` to explicitly use CPU
- CPU version works without PyTorch/PyTorch3D

## Usage

Once installed, run GPU-accelerated SVF computation:

```bash
python scripts/compute_svf_streets_gpu.py \
    --stl data/riodaspedras/raw/full_scan.stl \
    --roads data/riodaspedras/raw/roads_riodaspedras.shp \
    --footprints data/riodaspedras/raw/riodaspedras_buildings.shp \
    --area riodaspedras \
    --use-gpu \
    --spacing 3.0 \
    --sky-patches 145
```
