# Notebooks

This directory contains Jupyter notebooks for interactive data exploration and analysis.

## Available Notebooks

### `explore_favelas.ipynb`

Interactive exploration and selection interface for favelas in the Rio de Janeiro dataset.

**Features:**
- Load and visualize 1,074 favela boundaries
- Pre-compute statistics (building count, street length) using spatial indexing
- Interactive map with folium showing favela locations
- Searchable table with filtering capabilities
- Selection interface with checkboxes
- Export selected favelas to JSON/CSV for batch extraction

**Usage:**

1. **Install dependencies** (if not already installed):
   ```bash
   pip install folium ipywidgets pyarrow
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

3. **Open the notebook**:
   - Navigate to `notebooks/explore_favelas.ipynb`
   - Run all cells (Cell → Run All)

4. **Use the interface**:
   - Explore favelas on the interactive map
   - Filter and search using the table widgets
   - Select favelas using checkboxes
   - Export selection for batch extraction

**First Run:**
- The first run will compute statistics for all favelas (may take 5-15 minutes)
- Statistics are cached to disk:
  - `data/RJ/.cache/favela_statistics.parquet` (if pyarrow is installed - recommended)
  - `data/RJ/.cache/favela_statistics.csv` (fallback if pyarrow not available)
- Subsequent runs will load cached statistics instantly
- **Note:** If you get a parquet import error, the notebook will automatically fallback to CSV format

**Output:**
- Selected favelas are exported to `data/RJ/selected_favelas/selected_favelas.json`
- This file can be used with the extraction script for batch processing

## Requirements

### Core (required)
- geopandas
- pandas
- numpy
- tqdm

### Interactive features (optional but recommended)
- folium (for interactive maps)
- ipywidgets (for interactive widgets)
- pyarrow (for parquet caching - faster, but CSV fallback available)

## Notes

- The notebook uses spatial indexing for efficient queries (essential for 2.36M buildings)
- Statistics are cached to avoid recomputation
- The map may take a moment to render with all 1,074 favelas
- Widgets are limited to 100 favelas at a time for performance
