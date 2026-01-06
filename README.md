# Favela Morphometric Analysis

A Python pipeline for calculating morphometric metrics from building footprints with height attributes. Designed for analyzing informal settlement geometry and urban morphology.

## Features

- **Automatic data discovery**: Finds and processes any geospatial file (`.gpkg`, `.geojson`, `.shp`) in the data directory
- **Flexible input formats**: Supports standard (`base_height`, `top_height`) or alternative (`base`, `altura`) column names
- **Comprehensive filtering**: Multiple filtering options for height, area, volume, and h/w ratio
- **Rich visualizations**: Generates thematic maps, statistical distributions, and scatter plots
- **Robust validation**: Validates data quality, CRS, and geometry before processing

## Installation

### Prerequisites

- Python 3.11+
- Conda (recommended) or virtual environment

### Setup

**Using Conda:**
```bash
conda create -n IVF python=3.11
conda activate IVF
pip install -r requirements.txt
```

**Using Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

1. **Prepare your data:**
   ```bash
   mkdir -p data/raw
   # Copy your building footprints file to data/raw/
   ```

2. **Run the analysis:**
   ```bash
   python scripts/calculate_metrics.py
   ```

3. **Check results:**
   - `outputs/buildings_with_metrics.gpkg` - Enhanced dataset
   - `outputs/summary_stats.csv` - Summary statistics
   - `outputs/maps/` - All visualization files

## Input Data Requirements

### File Format
- Supported: `.gpkg`, `.geojson`, or `.shp`
- The script automatically finds the first geospatial file in `data/raw/`

### Required Attributes

**Standard format:**
- `base_height`: Elevation of building base (meters)
- `top_height`: Elevation of building top (meters)

**Alternative format (automatically converted):**
- `base`: Base elevation (meters)
- `altura`: Relative building height (meters)

### Coordinate System
- Projected CRS required (UTM preferred) for accurate area calculations
- The script validates CRS before processing

## Metrics Calculated

The pipeline calculates 5 fundamental morphometric metrics:

1. **height**: Building height (top_height - base_height) in meters
2. **area**: Footprint area in m²
3. **volume**: Building volume (area × height) in m³
4. **perimeter**: Footprint perimeter in meters
5. **hw_ratio**: Street canyon ratio (height/width) - building height divided by building width

## Output Files

### Data Outputs
- `buildings_with_metrics.gpkg`: Enhanced dataset with all calculated metrics
- `summary_stats.csv`: Descriptive statistics (mean, std, min, max, quartiles)

### Visualizations
- `height_volume_maps.png`: Height and volume thematic maps
- `multi_panel_summary.png`: 2×2 grid showing all key metrics
- `statistical_distributions.png`: Histograms and box plots for all metrics
- `scatter_plots.png`: Relationships between metrics

## Configuration

Edit `src/config.py` to customize:

### Filtering Parameters
```python
MAX_FILTER_HEIGHT = 20.0      # Maximum building height (m)
MAX_FILTER_AREA = 500.0      # Maximum footprint area (m²)
MAX_FILTER_VOLUME = 3000.0   # Maximum building volume (m³)
MAX_FILTER_HW_RATIO = 100.0  # Maximum h/w ratio
HEIGHT_AREA_PERCENTILE = 99.0  # Percentile for height/area outlier filtering
```

### Visualization Settings
```python
DPI = 300                    # Output resolution
FIGURE_SIZE = (12, 8)        # Figure dimensions
COLORMAP_HEIGHT = "viridis"  # Colormap for height maps
COLORMAP_VOLUME = "plasma"   # Colormap for volume maps
```

Set any filter to `None` to disable it.

## Project Structure

```
IVF/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── claude.md                # AI context documentation
│
├── data/                    # NOT tracked in git
│   ├── raw/                 # Input data directory
│   └── README.md            # Data documentation
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── config.py            # Configuration settings
│   ├── metrics.py           # Metrics calculation & validation
│   └── visualize.py         # Visualization functions
│
├── scripts/                 # Executable scripts
│   └── calculate_metrics.py # Main analysis script
│
└── outputs/                 # NOT tracked in git
    ├── buildings_with_metrics.gpkg
    ├── summary_stats.csv
    └── maps/                 # Visualization outputs
```

## Filtering Pipeline

The pipeline applies filters in the following order:

1. **Height filter**: Removes buildings exceeding maximum height
2. **Metrics calculation**: Computes all morphometric metrics
3. **Area filter**: Removes buildings exceeding maximum area
4. **Volume filter**: Removes buildings exceeding maximum volume
5. **H/W ratio filter**: Removes buildings with extreme h/w ratios
6. **Height/area ratio filter**: Removes outliers using percentile-based method

## Dependencies

- `geopandas>=0.14.0` - Geospatial data handling
- `shapely>=2.0.0` - Geometric operations
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.7.0` - Visualization

## Usage Examples

### Basic Usage
```bash
python scripts/calculate_metrics.py
```

### Custom Configuration
1. Edit `src/config.py` to adjust filtering thresholds
2. Run the script as normal

## Data Validation

The pipeline automatically validates:
- Required columns (height attributes)
- Coordinate Reference System (CRS)
- Geometry validity
- Data quality (null values, invalid heights)

Warnings are logged for non-critical issues; critical errors stop processing.

## Contributing

This is a research project. For questions or issues, please contact the project maintainer.

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite appropriately.
