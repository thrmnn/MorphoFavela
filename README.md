# Favela Morphometric Analysis

A Python pipeline for calculating morphometric metrics from building footprints with height attributes. Designed for analyzing informal settlement geometry and urban morphology.

## Features

- **Automatic data discovery**: Finds and processes any geospatial file (`.gpkg`, `.geojson`, `.shp`) in the data directory
- **Flexible input formats**: Supports standard (`base_height`, `top_height`) or alternative (`base`, `altura`) column names
- **Comprehensive filtering**: Multiple filtering options for height, area, volume, and h/w ratio
- **Rich visualizations**: Generates thematic maps, statistical distributions, and scatter plots
- **Robust validation**: Validates data quality, CRS, and geometry before processing

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed project roadmap. **Current status**: Phase 1 (Basic Morphometric Analysis), Phase 2 (SVF & Solar Access), and Phase 2.6 (Sky Exposure Plane Analysis) are complete.

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
   
   **Basic morphometric analysis:**
   ```bash
   python scripts/calculate_metrics.py
   ```
   
   **Sky View Factor (SVF) computation:**
   ```bash
   python scripts/compute_svf.py --stl data/raw/full_scan.stl --footprints data/raw/vidigal_buildings.shp --grid-spacing 5.0 --height 0.5 --sky-patches 145
   ```
   
   **Solar Access computation:**
   ```bash
   python scripts/compute_solar_access.py --stl data/raw/full_scan.stl --footprints data/raw/vidigal_buildings.shp --grid-spacing 5.0 --height 0.5 --threshold 3.0
   ```
   
   **Sky Exposure Plane Exceedance analysis:**
   ```bash
   python scripts/analyze_sky_exposure.py --stl data/raw/full_scan.stl --footprints data/raw/vidigal_buildings.shp --angle 45.0 --base-height 7.5 --front-setback 5.0 --side-setback 3.0
   ```

3. **Check results:**
   - `outputs/buildings_with_metrics.gpkg` - Enhanced dataset
   - `outputs/summary_stats.csv` - Summary statistics
   - `outputs/svf/` - SVF computation results
   - `outputs/solar/` - Solar access computation results
   - `outputs/sky_exposure/` - Sky exposure plane exceedance analysis results
   - `outputs/maps/` - All visualization files

## Input Data Requirements

### Basic Morphometric Analysis

**File Format:**
- Supported: `.gpkg`, `.geojson`, or `.shp`
- The script automatically finds the first geospatial file in `data/raw/`

**Required Attributes:**

**Standard format:**
- `base_height`: Elevation of building base (meters)
- `top_height`: Elevation of building top (meters)

**Alternative format (automatically converted):**
- `base`: Base elevation (meters)
- `altura`: Relative building height (meters)

### Sky View Factor (SVF) and Solar Access Computation

**Required Files:**
- **STL mesh**: Combined 3D scene containing both terrain and buildings (`.stl` file)
- **Building footprints** (optional): Shapefile for masking building interiors from ground-level analysis

**Methodology**: 
- **SVF**: Computes Sky View Factor using a discretized hemispherical dome approach. The sky is divided into equal-area patches, and ray-casting is used to determine how much of the sky is visible from each ground point. Uses `pyviewfactor`-style geometric visibility testing.
- **Solar Access**: Computes hours of direct sunlight by casting rays toward sun positions for winter solstice. Uses `pvlib` for accurate solar position calculations.

Both analyses:
- Exclude building interiors (only compute for ground-level points)
- Use the same ground mask logic for consistency
- Support progress monitoring during computation

### Sky Exposure Plane Exceedance Analysis

**Required Files:**
- **STL mesh**: Combined 3D scene containing both terrain and buildings (`.stl` file)
- **Building footprints**: Shapefile with building footprint polygons

**Methodology**: 
The sky exposure plane is an environmental performance envelope that defines the maximum allowable built form based on:
- **Base height**: Minimum height allowed before sky plane applies (typically 6-9m for 2-3 floors)
- **Setbacks**: Front (5m) and side/rear (3m) setbacks from footprint boundaries
- **Sky plane angle**: Inclined plane rising from setback lines at specified angle (typically 45°)

The analysis quantifies how much built volume exceeds this envelope, evaluating environmental implications (solar access and ventilation), NOT legal code compliance.

**Parameters:**
- `--angle`: Sky exposure plane angle in degrees (default: 45.0°)
- `--base-height`: Base height before sky plane applies (default: 7.5m, range: 6-9m)
- `--front-setback`: Front setback distance (default: 5.0m)
- `--side-setback`: Side/rear setback distance (default: 3.0m)

### Coordinate System
- Projected CRS required (UTM preferred) for accurate area calculations
- The script validates CRS before processing

## Metrics Calculated

The pipeline calculates 6 fundamental morphometric metrics:

1. **height**: Building height (top_height - base_height) in meters
2. **area**: Footprint area in m²
3. **volume**: Building volume (area × height) in m³
4. **perimeter**: Footprint perimeter in meters
5. **hw_ratio**: Street canyon ratio (height/width) - building height divided by building width
6. **inter_building_distance**: Distance to nearest neighbor building (m) - minimum distance between building boundaries. Calculated using spatial indexing for efficient computation on large datasets.

## Output Files

### Data Outputs
- `buildings_with_metrics.gpkg`: Enhanced dataset with all calculated metrics
- `summary_stats.csv`: Descriptive statistics (mean, std, min, max, quartiles)

### Visualizations
- `height_volume_maps.png`: Height, volume, and inter-building distance thematic maps
- `multi_panel_summary.png`: Multi-panel grid showing all key metrics (2×3 when inter-building distance is available)
- `statistical_distributions.png`: Histograms and box plots for all metrics
- `scatter_plots.png`: Relationships between metrics including inter-building distance

### SVF Outputs (`outputs/svf/`)
- `svf.npy`: 2D NumPy array of SVF values (NaN for building points)
- `svf.csv`: CSV file with columns: x, y, svf (only ground points)
- `svf_heatmap.png`: Top-down SVF heatmap visualization (0-1 scale)
- `svf_histogram.png`: Histogram of SVF value distribution
- `ground_mask_debug.png`: Debug plot showing ground points and building footprints

### Solar Access Outputs (`outputs/solar/`)
- `solar_access_heatmap.png`: Hours of direct sunlight heatmap
- `solar_access_threshold.png`: Binary classification map (red: <threshold, green: ≥threshold)
- `ground_mask_debug.png`: Debug plot showing ground points and building footprints

### Sky Exposure Plane Exceedance Outputs (`outputs/sky_exposure/`)
- `exceedance_map.png`: Plan view map showing buildings colored by exceedance ratio (%)
- `section_1.png`, `section_2.png`, `section_3.png`: Vertical sections showing actual built form vs sky exposure plane envelope
- `exceedance_results.csv`: Detailed exceedance metrics per building (total volume, exceeding volume, exceedance ratio, max exceedance height)

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

### SVF and Solar Access Computation

Both scripts are configured via command-line arguments:

**SVF Script:**
- `--grid-spacing`: Grid resolution in meters (e.g., 5.0m)
- `--height`: Evaluation height above ground (e.g., 0.5m)
- `--sky-patches`: Number of sky patches for hemisphere discretization (e.g., 145 or 290)
- `--footprints`: Optional path to building footprints shapefile
- `--buffer-distance`: Buffer distance for building footprints (default: 0.25m)

**Solar Access Script:**
- `--grid-spacing`: Grid resolution in meters (e.g., 5.0m)
- `--height`: Evaluation height above ground (e.g., 0.5m)
- `--threshold`: Solar access threshold in hours (default: 2.0h)
- `--timestep`: Solar calculation time step in minutes (default: 60)
- `--latitude` / `--longitude`: Site coordinates (default: Rio de Janeiro)
- `--footprints`: Optional path to building footprints shapefile
- `--buffer-distance`: Buffer distance for building footprints (default: 0.25m)

See `python scripts/compute_svf.py --help` and `python scripts/compute_solar_access.py --help` for all options.

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
│   ├── visualize.py         # Visualization functions
│   └── svf_utils.py         # Shared utilities for SVF and solar access
│
├── scripts/                 # Executable scripts
│   ├── calculate_metrics.py # Basic morphometric analysis
│   ├── compute_svf.py      # Sky View Factor computation
│   ├── compute_solar_access.py  # Solar access computation
│   └── analyze_sky_exposure.py  # Sky exposure plane exceedance analysis
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
- `tqdm>=4.65.0` - Progress bars
- `pyviewfactor>=1.0.0` - View factor computation (for SVF)
- `pyvista>=0.32.0` - 3D geometry handling and ray tracing
- `pvlib>=0.10.0` - Solar position calculations (for solar access)

## Usage Examples

### Basic Morphometric Analysis
```bash
python scripts/calculate_metrics.py
```

### SVF Computation
```bash
python scripts/compute_svf.py --stl data/raw/full_scan.stl --footprints data/raw/vidigal_buildings.shp --grid-spacing 5.0 --height 0.5 --sky-patches 145
```

### Solar Access Computation
```bash
python scripts/compute_solar_access.py --stl data/raw/full_scan.stl --footprints data/raw/vidigal_buildings.shp --grid-spacing 5.0 --height 0.5 --threshold 3.0
```

### Sky Exposure Plane Exceedance Analysis
```bash
python scripts/analyze_sky_exposure.py --stl data/raw/full_scan.stl --footprints data/raw/vidigal_buildings.shp --angle 45.0 --base-height 7.5 --front-setback 5.0 --side-setback 3.0
```

### Custom Configuration
1. Edit `src/config.py` to adjust filtering thresholds for morphometric analysis
2. Use command-line arguments to configure SVF and solar access parameters
3. Run the scripts as normal

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
