# Urban Morphology Metrics Implementation Plan

**Status**: Planned (not yet implemented)
**Phase**: 4
**Last Updated**: 2026-01-22

## Executive Summary

This plan details the implementation of urban morphology metrics for the IVF (Favela Morphometric Analysis) project. The new features include plan area density (lambda_p), frontal area density (lambda_f), height variability (sigma_h), street orientation entropy (H), typology clustering, and zone flagging.

---

## 1. File Structure

### New Files to Create

```
scripts/
  compute_urban_morphology.py      # Main unified script for morphology metrics
  compute_typology_clustering.py   # K-means/hierarchical clustering on morphology

src/
  urban_morphology.py              # Core morphology metric functions
```

### Files to Modify

```
scripts/run_area_analyses.py       # Add morphology analysis step
scripts/compare_areas.py           # Add morphology comparison
src/config.py                      # Add ANALYSIS_TYPES entry
```

---

## 2. Core Module: `src/urban_morphology.py`

### Key Functions and Signatures

```python
"""Urban morphology metrics calculation module."""

import numpy as np
import geopandas as gpd
from shapely.geometry import box, LineString
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def generate_analysis_units(
    footprints: gpd.GeoDataFrame,
    grid_size: float = 50.0
) -> gpd.GeoDataFrame:
    """
    Generate 50m x 50m analysis units covering the study area.

    Args:
        footprints: Building footprints GeoDataFrame
        grid_size: Size of grid cells in meters (default: 50m)

    Returns:
        GeoDataFrame with grid cell polygons as analysis units
    """


def compute_plan_area_density(
    analysis_units: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame
) -> np.ndarray:
    """
    Compute plan area density (lambda_p) for each analysis unit.

    lambda_p = A_footprint / A_total

    Args:
        analysis_units: GeoDataFrame with analysis unit polygons
        footprints: GeoDataFrame with building footprints

    Returns:
        Array of lambda_p values (0-1) for each unit
    """


def compute_frontal_area_density(
    analysis_units: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    wind_directions: int = 8,
    default_height: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frontal area density (lambda_f) for each analysis unit.

    lambda_f = A_frontal / A_total

    Args:
        analysis_units: GeoDataFrame with analysis unit polygons
        footprints: GeoDataFrame with building footprints (must have 'height' column)
        wind_directions: Number of wind directions (8 or 16)
        default_height: Default height if 'height' column missing

    Returns:
        Tuple of (mean_lambda_f, directional_lambda_f_array)
    """


def compute_height_variability(
    analysis_units: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame
) -> np.ndarray:
    """
    Compute height variability (sigma_h) within each analysis unit.

    sigma_h = std(building_heights) within unit

    Args:
        analysis_units: GeoDataFrame with analysis unit polygons
        footprints: GeoDataFrame with building footprints (must have 'height' column)

    Returns:
        Array of sigma_h values (meters) for each unit
    """


def compute_street_orientation_entropy(
    streets: gpd.GeoDataFrame,
    n_bins: int = 36
) -> Tuple[float, np.ndarray]:
    """
    Compute street orientation entropy at study area level.

    H = -sum(p_i * log(p_i)) for i in direction bins

    High entropy = irregular pattern (like favelas)
    Low entropy = grid pattern (like formal areas)

    Args:
        streets: GeoDataFrame with LineString geometries
        n_bins: Number of direction bins (default: 36 = 10 degrees each)

    Returns:
        Tuple of (entropy_value, orientation_histogram)
    """


def compute_street_orientation_entropy_per_unit(
    analysis_units: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    n_bins: int = 36
) -> np.ndarray:
    """
    Compute street orientation entropy for each analysis unit.
    """


def aggregate_morphology_metrics(
    analysis_units: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame = None,
    svf_raster: np.ndarray = None,
    svf_extent: Tuple[float, float, float, float] = None
) -> gpd.GeoDataFrame:
    """
    Compute all morphology metrics and aggregate to analysis units.

    Adds columns:
        - lambda_p: Plan area density
        - lambda_f: Frontal area density (mean across directions)
        - sigma_h: Height variability
        - entropy_h: Street orientation entropy (if streets provided)
        - mean_svf: Mean SVF (if SVF raster provided)
    """


def flag_zones(
    units_with_metrics: gpd.GeoDataFrame,
    svf_threshold: float = 0.3,
    lambda_f_threshold: float = 0.4
) -> gpd.GeoDataFrame:
    """
    Flag analysis units meeting proxy thresholds.

    Adds columns:
        - flag_low_svf: SVF < 0.3 (poor sky visibility)
        - flag_high_density: lambda_f > 0.4 (high frontal density)
        - flag_compound: Both conditions met
    """
```

---

## 3. Algorithm Details

### 3.1 Plan Area Density (lambda_p)

```
For each analysis unit:
    1. Find buildings that intersect the unit
    2. Clip building footprints to unit boundary
    3. Sum clipped footprint areas
    4. lambda_p = sum(clipped_areas) / unit_area
    5. Clamp to [0, 1]
```

**Complexity**: O(n_units * m_buildings) with spatial indexing

### 3.2 Frontal Area Density (lambda_f)

```
Wind directions: [0, 45, 90, 135, 180, 225, 270, 315] degrees (8 directions)

For each analysis unit:
    1. Find buildings within unit
    2. For each building:
        a. Get building height (from 'height' column or default)
        b. Extract perimeter as series of line segments
        c. For each wind direction:
            - For each segment:
                - Compute segment normal angle
                - angle_diff = |segment_normal - wind_direction|
                - projected_length = segment_length * cos(angle_diff) (if facing wind)
                - frontal_area += projected_length * building_height
    3. For each direction:
        lambda_f[dir] = sum(frontal_areas) / unit_area
    4. mean_lambda_f = mean(lambda_f across all directions)
```

**Simplified Approach** (recommended for initial implementation):
```
For each building:
    frontal_area = perimeter * height / 4
    (Approximation: assumes square building with 1/4 of perimeter facing each cardinal direction)
```

### 3.3 Height Variability (sigma_h)

```
For each analysis unit:
    1. Find buildings with centroid within unit
    2. Extract height values
    3. sigma_h = np.std(heights)
    4. If < 2 buildings, sigma_h = 0
```

### 3.4 Street Orientation Entropy

```
1. For each street segment:
    a. Extract coordinates
    b. For each sub-segment (point pairs):
        - Compute angle: theta = atan2(dy, dx)
        - Normalize to [0, 180): theta = theta % 180
        - Accumulate length in corresponding bin

2. Compute probability distribution:
    p_i = length_in_bin_i / total_length

3. Compute Shannon entropy:
    H = -sum(p_i * log2(p_i)) for p_i > 0

4. Normalize to [0, 1]:
    H_normalized = H / log2(n_bins)
```

### 3.5 Zone Flagging

```
For each analysis unit:
    flag_low_svf = (mean_svf < 0.3)
    flag_high_density = (lambda_f > 0.4)
    flag_compound = flag_low_svf AND flag_high_density
```

---

## 4. Main Script: `scripts/compute_urban_morphology.py`

```python
#!/usr/bin/env python3
"""
Urban Morphology Metrics Computation

Computes urban morphology metrics for environmental analysis:
- Plan area density (lambda_p)
- Frontal area density (lambda_f)
- Height variability (sigma_h)
- Street orientation entropy

Usage:
    python scripts/compute_urban_morphology.py --area vidigal
    python scripts/compute_urban_morphology.py --area copacabana --grid-size 50.0
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    get_area_data_dir,
    get_area_analysis_dir,
    SUPPORTED_AREAS
)
from src.urban_morphology import (
    generate_analysis_units,
    compute_plan_area_density,
    compute_frontal_area_density,
    compute_height_variability,
    compute_street_orientation_entropy,
    compute_street_orientation_entropy_per_unit,
    aggregate_morphology_metrics,
    flag_zones
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Compute urban morphology metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--area', type=str, required=True,
                       choices=SUPPORTED_AREAS,
                       help='Area name (vidigal, copacabana)')
    parser.add_argument('--grid-size', type=float, default=50.0,
                       help='Analysis unit grid size in meters (default: 50.0)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: outputs/<area>/morphology)')

    args = parser.parse_args()

    # Implementation...


if __name__ == "__main__":
    main()
```

---

## 5. Output Specifications

### GeoPackage Output: `morphology_metrics.gpkg`

| Column | Type | Description |
|--------|------|-------------|
| unit_id | int | Analysis unit identifier |
| geometry | Polygon | 50m x 50m grid cell |
| lambda_p | float | Plan area density (0-1) |
| lambda_f | float | Frontal area density (mean, 0-1) |
| sigma_h | float | Height variability (meters) |
| entropy_h | float | Street orientation entropy (0-1) |
| mean_svf | float | Mean SVF within unit (0-1) |
| n_buildings | int | Number of buildings in unit |
| total_area | float | Unit area (m^2) |
| built_area | float | Total footprint area (m^2) |
| flag_low_svf | bool | SVF < 0.3 flag |
| flag_high_density | bool | lambda_f > 0.4 flag |
| flag_compound | bool | Both flags true |

### Visualization Outputs

- `lambda_p_map.png` - Choropleth of plan area density
- `lambda_f_map.png` - Choropleth of frontal area density
- `sigma_h_map.png` - Choropleth of height variability
- `entropy_map.png` - Choropleth of street orientation entropy
- `flagged_zones_map.png` - Map highlighting flagged analysis units
- `metrics_distributions.png` - Histograms of all metrics
- `metrics_correlation.png` - Correlation matrix heatmap

---

## 6. Integration Points

### 6.1 `src/config.py` Modification

Add to `ANALYSIS_TYPES`:

```python
ANALYSIS_TYPES = {
    # ... existing types ...
    "morphology": "morphology",         # Urban morphology metrics
    "typology": "typology",             # Morphological typology clustering
}
```

### 6.2 `scripts/run_area_analyses.py` Modification

Add morphology computation step after metrics:

```python
# 8. Urban Morphology Metrics
morphology_output = get_area_analysis_dir(area, "morphology")
cmd = [
    sys.executable, "scripts/compute_urban_morphology.py",
    "--area", area,
    "--grid-size", "50.0",
    "--output-dir", str(morphology_output)
]
if not run_command(cmd, "Urban Morphology Metrics"):
    print("Warning: Morphology computation failed, continuing...")
```

### 6.3 `scripts/compare_areas.py` Modification

Add morphology comparison function:

```python
def compare_morphology_metrics(vidigal_data: dict, copacabana_data: dict) -> dict:
    """Compare morphology metrics between areas."""
    # Load morphology data
    # Compare distributions of lambda_p, lambda_f, sigma_h, entropy_h
    # Statistical tests (Mann-Whitney U)
    # Effect size calculations
    # Return comparison dict
```

---

## 7. Typology Clustering

### Script: `scripts/compute_typology_clustering.py`

Cluster analysis units into morphological typologies using K-means or hierarchical clustering on (lambda_p, lambda_f, sigma_h, entropy_h).

**Features**:
- Standardized feature scaling
- K-means and hierarchical clustering options
- Silhouette score evaluation
- Typology map visualization
- Parallel coordinates plot
- Radar chart of cluster centers
- Dendrogram (for hierarchical)

**Usage**:
```bash
python scripts/compute_typology_clustering.py --area vidigal --n-clusters 5
python scripts/compute_typology_clustering.py --areas vidigal copacabana --method hierarchical
```

---

## 8. Data Flow Diagram

```
Input Data:
    ├── Building Footprints (GeoPackage/Shapefile)
    │   └── Columns: geometry, height (or base_height, top_height)
    │
    ├── Street Network (GeoPackage/Shapefile)
    │   └── Columns: geometry (LineString)
    │
    └── SVF Raster (NumPy NPY) [optional]

                    ↓

    ┌─────────────────────────────────┐
    │  generate_analysis_units()       │
    │  - Create 50m x 50m grid        │
    └─────────────────────────────────┘
                    ↓

    ┌─────────────────────────────────┐
    │  For each analysis unit:        │
    │                                 │
    │  ├── compute_plan_area_density()│
    │  │   λp = A_footprint / A_total │
    │  │                              │
    │  ├── compute_frontal_area_density()
    │  │   λf = A_frontal / A_total   │
    │  │                              │
    │  ├── compute_height_variability()│
    │  │   σh = std(heights)          │
    │  │                              │
    │  └── compute_street_entropy()   │
    │      H = -Σ p_i log(p_i)        │
    └─────────────────────────────────┘
                    ↓

    ┌─────────────────────────────────┐
    │  flag_zones()                   │
    │  - SVF < 0.3                    │
    │  - λf > 0.4                     │
    └─────────────────────────────────┘
                    ↓

Output Files:
    ├── morphology_metrics.gpkg
    ├── morphology_metrics.csv
    └── Visualizations (PNG files)
```

---

## 9. Implementation Sequence

1. **Phase 1: Core Functions** (src/urban_morphology.py)
   - `generate_analysis_units()`
   - `compute_plan_area_density()`
   - `compute_height_variability()`
   - Unit tests for each function

2. **Phase 2: Complex Metrics**
   - `compute_frontal_area_density()` (start with simplified approach)
   - `compute_street_orientation_entropy()`
   - Unit tests

3. **Phase 3: Main Script**
   - `scripts/compute_urban_morphology.py`
   - Visualization functions
   - Integration with config.py

4. **Phase 4: Zone Flagging & Integration**
   - `flag_zones()` implementation
   - Modify `run_area_analyses.py`
   - Run on both study areas

5. **Phase 5: Typology Clustering**
   - `scripts/compute_typology_clustering.py`
   - Multi-area clustering support

6. **Phase 6: Comparative Analysis**
   - Modify `compare_areas.py`
   - Add morphology to PDF report

---

## 10. Testing Approach

### Unit Tests

```python
# tests/test_urban_morphology.py

def test_plan_area_density_empty_unit():
    """Test lambda_p = 0 for empty unit."""

def test_plan_area_density_full_coverage():
    """Test lambda_p = 1 for full coverage."""

def test_street_entropy_grid():
    """Test low entropy for grid pattern."""

def test_street_entropy_random():
    """Test high entropy for random orientations."""
```

### Integration Tests

```python
def test_full_morphology_pipeline():
    """Test complete morphology computation pipeline."""

def test_typology_clustering():
    """Test typology clustering produces valid results."""
```

---

## 11. Potential Challenges

1. **Building Height Data**: Some footprints may lack height attributes
   - Solution: Use metrics output (`buildings_with_metrics.gpkg`) which has computed heights

2. **Street Network Coverage**: Streets may not cover all analysis units
   - Solution: Mark units without streets as having undefined entropy

3. **Coordinate System Alignment**: Streets and footprints may be in different CRS
   - Solution: Follow existing pattern in `svf_utils.py` for coordinate transformation

4. **Memory for Large Areas**: Computing metrics for thousands of units
   - Solution: Use spatial indexing (R-tree) and process in batches with tqdm

5. **Edge Effects**: Units at study area boundary may have incomplete data
   - Solution: Flag boundary units or require minimum coverage threshold

---

## 12. Critical Files for Implementation Reference

- `/home/theo/IVF/scripts/compute_occupancy_density.py` - Pattern for unit-level aggregation, grid generation, GeoPackage output
- `/home/theo/IVF/src/metrics.py` - Building height extraction, inter-building distance patterns
- `/home/theo/IVF/scripts/run_area_analyses.py` - Integration point for batch processing
- `/home/theo/IVF/scripts/compare_areas.py` - Pattern for comparative analysis across areas
- `/home/theo/IVF/src/config.py` - Configuration patterns and area management to extend
