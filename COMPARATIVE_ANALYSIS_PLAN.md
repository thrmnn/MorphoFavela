# Comparative Analysis Plan: Formal vs Informal Settlements

## Overview

This document outlines the approach for comparing Vidigal (informal settlement) and Copacabana (formal neighborhood) across all computed metrics and analyses.

## Analysis Framework

### 1. Data Loading and Preparation

**Load metrics data:**
```python
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# Load building metrics
vidigal_buildings = gpd.read_file("outputs/vidigal/metrics/buildings_with_metrics.gpkg")
copacabana_buildings = gpd.read_file("outputs/copacabana/metrics/buildings_with_metrics.gpkg")

# Add area labels
vidigal_buildings['area_type'] = 'informal'
copacabana_buildings['area_type'] = 'formal'
```

**Load raster data:**
- SVF rasters (`svf.npy`)
- Solar access rasters (`solar_access.npy`)
- Porosity rasters (`porosity.npy`)
- Deprivation index rasters (`hotspot_index.npy`)

**Load summary statistics:**
- CSV files with summary stats for each area

## 2. Morphometric Metrics Comparison

### 2.1 Statistical Summary Comparison

**Create comparison table:**
- Mean, median, std, min, max for each metric
- Calculate differences and ratios between formal/informal
- Identify which metrics show largest differences

**Metrics to compare:**
- Building height (mean, distribution)
- Footprint area
- Building volume
- Height/width ratio (H/W)
- Inter-building distance
- Building count and density

**Output:** `comparative/tables/comparison_morphometric_stats.csv`

### 2.2 Distribution Comparisons

**Side-by-side visualizations:**
1. **Histogram overlays** - Show both distributions on same axes
2. **Box plot comparisons** - Formal vs Informal side-by-side
3. **Kernel density estimates** - Smoothed distribution comparison
4. **Quantile-quantile (Q-Q) plots** - Compare distribution shapes

**Metrics to visualize:**
- Height distributions
- Area distributions
- Volume distributions
- H/W ratio distributions
- Inter-building distance distributions

**Statistical tests:**
- Kolmogorov-Smirnov test (distribution differences)
- Mann-Whitney U test (median differences)
- t-test (if normally distributed)

**Output:** `comparative/visualizations/comparison_morphometric_distributions.png`

### 2.3 Scatter Plot Comparisons

**Compare relationships:**
- Height vs Area (formal vs informal)
- Volume vs Area
- Height vs Inter-building distance
- Different color/symbol for each area type

**Output:** `comparative/visualizations/comparison_morphometric_scatter.png`

## 3. Environmental Performance Comparison

### 3.1 SVF (Sky View Factor) Comparison

**Raster comparison:**
1. Load both SVF rasters
2. Calculate statistics for each:
   - Mean, median, percentiles (10th, 25th, 50th, 75th, 90th)
   - Fraction of area with SVF < 0.5 (low sky visibility)
   - Fraction of area with SVF > 0.7 (high sky visibility)

**Side-by-side visualizations:**
1. **Heatmap comparison** - Two panels side-by-side
2. **Histogram comparison** - Distribution overlay
3. **Difference map** - (Copacabana SVF - Vidigal SVF) if aligned

**Spatial pattern analysis:**
- Identify areas with very low SVF in each settlement type
- Compare spatial distribution of low SVF zones

**Output:**
- `comparative/tables/comparison_svf_stats.csv`
- `comparative/visualizations/comparison_svf_side_by_side.png`
- `comparative/visualizations/comparison_svf_distributions.png`

### 3.2 Solar Access Comparison

**Statistics to compare:**
- Mean hours of direct sunlight (winter solstice)
- Fraction of area with < 2 hours (solar deficit)
- Fraction of area with < 3 hours
- Fraction of area with ≥ 3 hours (acceptable)

**Visualizations:**
1. Side-by-side heatmaps
2. Distribution comparison (histogram overlay)
3. Threshold comparison (deficit vs acceptable)

**Output:**
- `comparative/tables/comparison_solar_stats.csv`
- `comparative/visualizations/comparison_solar_side_by_side.png`

### 3.3 Porosity Comparison

**Statistics:**
- Mean sectional porosity
- Distribution of porosity values
- Fraction of area with low porosity (< 0.3 = dense)

**Visualizations:**
- Side-by-side porosity maps
- Distribution comparison

**Output:**
- `comparative/tables/comparison_porosity_stats.csv`
- `comparative/visualizations/comparison_porosity_side_by_side.png`

## 4. Density and Occupancy Comparison

### 4.1 Occupancy Density Proxy

**Compare:**
- Mean density (built volume / open space)
- Distribution of density values
- Fraction of area with high density (top 10th percentile)

**Visualizations:**
- Side-by-side density maps
- Distribution comparison

**Output:**
- `comparative/tables/comparison_density_stats.csv`
- `comparative/visualizations/comparison_density_side_by_side.png`

## 5. Deprivation Index Comparison

### 5.1 Hotspot Analysis

**Compare deprivation indices:**
- Mean deprivation index
- Fraction of area classified as:
  - Extreme hotspots (top 10%)
  - High deprivation (10-20%)
  - Baseline (bottom 80%)

**Visualizations:**
1. **Side-by-side hotspot maps** - Classified deprivation
2. **Continuous comparison** - Side-by-side continuous heatmaps
3. **Distribution comparison** - Histogram overlay

**Key questions:**
- Which area has more extreme hotspots?
- Are hotspots more concentrated or dispersed?
- What are the spatial patterns?

**Output:**
- `comparative/tables/comparison_deprivation_stats.csv`
- `comparative/visualizations/comparison_deprivation_hotspots.png`
- `comparative/visualizations/comparison_deprivation_distributions.png`

## 6. Sky Exposure Plane Exceedance Comparison

### 6.1 Exceedance Patterns

**Compare:**
- Mean exceedance ratio per building
- Number/fraction of buildings exceeding envelope
- Distribution of exceedance values
- Maximum exceedance in each area

**Visualizations:**
- Side-by-side exceedance maps
- Distribution comparison

**Output:**
- `comparative/tables/comparison_sky_exposure_stats.csv`
- `comparative/visualizations/comparison_sky_exposure_side_by_side.png`

## 7. Integrated Comparison Summary

### 7.1 Multi-Metric Dashboard

**Create comprehensive comparison visualization:**
- Grid of small multiples showing all metrics
- Color-coded by area type (formal/informal)
- Highlight key differences

**Output:** `comparative/visualizations/comparison_dashboard.png`

### 7.2 Summary Statistics Table

**Single comprehensive table with:**
- All key metrics for both areas
- Differences (formal - informal)
- Ratios (formal / informal)
- Significance indicators

**Output:** `comparative/tables/comparison_comprehensive_summary.csv`

## 8. Key Research Questions to Answer

1. **Morphology:**
   - How do building heights differ between formal and informal?
   - Are informal settlements more compact (smaller footprints)?
   - Do formal settlements have better spacing (inter-building distance)?

2. **Environmental Performance:**
   - Which area has better sky visibility (SVF)?
   - Which area has better solar access?
   - Which area has better ventilation potential (porosity)?

3. **Environmental Deprivation:**
   - Which area has more environmental hotspots?
   - Are deprivation patterns different in formal vs informal?
   - What are the compounding factors?

4. **Planning Implications:**
   - What can formal settlements learn from informal?
   - What interventions are most needed in each type?

## Implementation Structure

### Script: `scripts/compare_areas.py`

**Main workflow:**
```python
def main():
    # 1. Load all data
    vidigal_data = load_area_data('vidigal')
    copacabana_data = load_area_data('copacabana')
    
    # 2. Compare morphometric metrics
    compare_morphometric_metrics(vidigal_data, copacabana_data)
    
    # 3. Compare environmental performance
    compare_environmental_metrics(vidigal_data, copacabana_data)
    
    # 4. Compare deprivation indices
    compare_deprivation_indices(vidigal_data, copacabana_data)
    
    # 5. Generate summary reports
    generate_comparison_summary(vidigal_data, copacabana_data)
```

### Output Directory Structure

```
outputs/comparative/
├── metrics/
│   ├── comparison_morphometric_stats.csv
│   ├── comparison_environmental_stats.csv
│   └── comparison_comprehensive_summary.csv
├── visualizations/
│   ├── comparison_morphometric_distributions.png
│   ├── comparison_morphometric_scatter.png
│   ├── comparison_svf_side_by_side.png
│   ├── comparison_solar_side_by_side.png
│   ├── comparison_porosity_side_by_side.png
│   ├── comparison_density_side_by_side.png
│   ├── comparison_deprivation_hotspots.png
│   ├── comparison_sky_exposure_side_by_side.png
│   └── comparison_dashboard.png
└── tables/
    └── (detailed comparison tables)
```

## Statistical Methods

### Descriptive Statistics
- Mean, median, standard deviation
- Percentiles (10th, 25th, 50th, 75th, 90th)
- Min, max, range

### Inferential Statistics (where appropriate)
- **t-test** or **Mann-Whitney U** for comparing means/medians
- **Kolmogorov-Smirnov test** for comparing distributions
- **Effect sizes** (Cohen's d) to quantify differences

### Visualization Approaches
- **Overlaid histograms** - Show both distributions
- **Box plots** - Side-by-side comparison
- **Violin plots** - Show distribution shape and density
- **Side-by-side heatmaps** - For spatial comparisons
- **Difference maps** - If rasters are aligned

## Notes on Interpretation

1. **Filtering differences**: Remember that Vidigal data is filtered (informal), while Copacabana is not (formal). This may affect comparisons.

2. **Scale differences**: Areas may have different spatial extents. Consider normalizing by area or using density measures.

3. **Context**: Formal vs informal differences reflect different:
   - Development processes
   - Regulatory frameworks
   - Socioeconomic contexts
   - Historical trajectories

4. **Causality**: Differences are descriptive, not causal. Formal vs informal classification is a proxy for multiple underlying factors.

## Expected Insights

Based on typical patterns:

**Formal settlements (Copacabana) might show:**
- Larger, taller buildings
- More uniform building sizes
- Better spacing (larger inter-building distances)
- Potentially better environmental performance (planned design)

**Informal settlements (Vidigal) might show:**
- More compact, smaller buildings
- Greater variation in building sizes
- Tighter spacing (smaller inter-building distances)
- Different environmental performance patterns

But the analysis will reveal actual patterns!

