# Favela Morphometric Analysis - AI Context

## Project Overview
Python pipeline for calculating morphometric metrics from building footprints with height attributes. Designed for analyzing informal settlement geometry.

## Key Metrics
- **height**: Building height (m)
- **area**: Footprint area (m²)
- **volume**: Building volume (m³)
- **perimeter**: Footprint perimeter (m)
- **hw_ratio**: Street canyon ratio (height/width)

## Data Input
- Formats: `.gpkg`, `.geojson`, `.shp`
- Columns: `base_height`/`top_height` OR `base`/`altura` (auto-normalized)
- CRS: Projected (UTM preferred)

## Filtering Pipeline
1. Height filter (max 20m)
2. Metrics calculation
3. Area filter (max 500m²)
4. Volume filter (max 3000m³)
5. H/W ratio filter (max 100)
6. Height/area ratio outlier filter (99th percentile)

## Configuration
All parameters in `src/config.py`:
- Filtering thresholds
- Visualization settings
- Data paths

## Code Structure
- `src/metrics.py`: Metrics calculation & validation
- `src/visualize.py`: Visualization functions
- `src/config.py`: Configuration
- `scripts/calculate_metrics.py`: Main pipeline

## Notes
- Uses logging, not print statements
- Type hints and docstrings throughout
- Data validation before processing
- Percentile-based outlier filtering for h/w ratio visualization