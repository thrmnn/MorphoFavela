# Favela Data Extraction Plan

## Overview
Extract and prepare data for individual favelas from the city-wide Rio de Janeiro dataset (`data/RJ/`). The dataset contains:
- **2.36M buildings** (`buildings_RJ_2019.shp`) - whole city
- **134K street segments** (`Logradouros.shp`) - whole city  
- **1,074 favelas** (`Favelas_Limit_2019.shp`) - favela boundaries

## Goals
1. Create an interactive notebook for exploring and selecting favelas
2. Extract building footprints, street networks, and boundaries for selected favelas
3. Organize extracted data into `data/{favela_name}/raw/` structure
4. Ensure computational efficiency for processing large datasets
5. Support both manual selection and batch extraction

## Data Structure

### Input (data/RJ/)
```
data/RJ/
├── buildings_RJ_2019.shp      # 2.36M buildings (altura, base columns)
├── Favelas_Limit_2019.shp     # 1,074 favela boundaries
└── Logradouros.shp            # 134K street segments
```

### Output Structure (per favela)
```
data/{favela_name}/
├── raw/
│   ├── {favela_name}_buildings.shp    # Extracted buildings
│   ├── roads_{favela_name}.shp        # Extracted streets
│   └── {favela_name}_boundary.shp     # Favela boundary (optional)
└── README.md                           # Favela metadata
```

## Implementation Plan

### Phase 1: Interactive Exploration Notebook
**File**: `notebooks/explore_favelas.ipynb`

**Features**:
- Load and display favela boundaries
- Interactive map with folium/ipyleaflet showing:
  - Favela boundaries (colored by area/population)
  - Building density overlay
  - Street network visualization
- Searchable table with favela metadata:
  - Name, area, population, neighborhood, etc.
  - Building count (pre-computed)
  - Street length (pre-computed)
- Selection interface:
  - Checkboxes for individual favelas
  - Filter by area, population, neighborhood
  - Export selected list to JSON/CSV

**Computation Strategy**:
- Pre-compute statistics once (cache results):
  - Building count per favela
  - Street length per favela
  - Area calculations
- Use spatial index (R-tree) for fast queries
- Lazy loading for visualization (only show selected favelas)

### Phase 2: Extraction Script
**File**: `scripts/extract_favela_data.py`

**Core Functions**:

#### 2.1 Spatial Indexing (Efficiency)
```python
def build_spatial_index(gdf: gpd.GeoDataFrame) -> rtree.index.Index
```
- Build R-tree spatial index for buildings and streets
- Enables fast spatial queries (O(log n) instead of O(n))
- Critical for 2.36M buildings

#### 2.2 Building Extraction
```python
def extract_buildings(
    favela_boundary: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    spatial_index: rtree.index.Index,
    buffer: float = 0.0
) -> gpd.GeoDataFrame
```
- Use spatial index to find candidate buildings
- Filter with `within()` or `intersects()` (with optional buffer)
- Preserve all original columns
- Validate CRS consistency

#### 2.3 Street Extraction
```python
def extract_streets(
    favela_boundary: gpd.GeoDataFrame,
    streets_gdf: gpd.GeoDataFrame,
    spatial_index: rtree.index.Index,
    buffer: float = 50.0  # Include nearby streets
) -> gpd.GeoDataFrame
```
- Extract streets within/buffered around favela boundary
- Buffer recommended (50m) to include connecting streets
- Filter by intersection percentage (e.g., >10% of street within boundary)

#### 2.4 Batch Processing
```python
def extract_multiple_favelas(
    favela_names: list[str],
    output_base_dir: Path,
    buildings_gdf: gpd.GeoDataFrame,
    streets_gdf: gpd.GeoDataFrame,
    favelas_gdf: gpd.GeoDataFrame,
    parallel: bool = True,
    n_workers: int = 4
) -> dict[str, dict]
```
- Process multiple favelas in parallel
- Use multiprocessing for CPU-bound operations
- Progress tracking with tqdm
- Error handling per favela (continue on failure)

### Phase 3: Configuration & Metadata

#### 3.1 Configuration File
**File**: `src/favela_config.py`
```python
# Extraction parameters
EXTRACTION_BUFFER_BUILDINGS = 20.0  # meters
EXTRACTION_BUFFER_STREETS = 20.0   # meters
MIN_STREET_INTERSECTION = 0.3      # 30% of street must be within boundary

# Output settings
OUTPUT_FORMAT = 'shp'  # or 'gpkg'
NORMALIZE_NAMES = True  # Sanitize favela names for file paths
```

#### 3.2 Metadata Generation
- Create README.md per favela with:
  - Original name, code, neighborhood
  - Population, area statistics
  - Building count, street length
  - Extraction date, source data info

### Phase 4: Validation & Quality Checks

#### 4.1 Data Validation
- CRS consistency check
- Geometry validity
- Non-empty results (warn if no buildings/streets)
- Column presence check

#### 4.2 Quality Metrics
- Building density (buildings/km²)
- Street density (km/km²)
- Coverage ratio (extracted area / favela area)

## Performance Optimizations

### 1. Spatial Indexing
- **R-tree index**: Use `rtree` or `geopandas.sindex`
- Build once, reuse for all favelas
- Reduces query time from O(n) to O(log n)

### 2. Chunked Processing
- Process buildings in chunks (e.g., 100K at a time)
- Memory-efficient for large datasets
- Use `geopandas.read_file()` with chunksize parameter

### 3. Parallel Processing
- Use `multiprocessing` or `joblib` for batch extraction
- Parallelize across favelas (not within single favela)
- Limit workers based on I/O constraints

### 4. Caching
- Cache spatial indexes to disk (pickle)
- Cache pre-computed statistics
- Avoid recomputing for same favela

### 5. Memory Management
- Use `geopandas.read_file()` with bbox parameter when possible
- Clear intermediate results
- Process and write immediately (don't accumulate in memory)

## Implementation Steps

### Step 1: Create Notebook Structure
- [ ] Set up notebook with data loading
- [ ] Add interactive map visualization
- [ ] Create favela selection interface
- [ ] Add statistics display

### Step 2: Implement Core Extraction Functions
- [ ] Build spatial indexing utilities
- [ ] Implement building extraction with spatial index
- [ ] Implement street extraction with buffer
- [ ] Add validation functions

### Step 3: Create Batch Processing Script
- [ ] Command-line interface
- [ ] Single favela extraction
- [ ] Batch extraction with parallelization
- [ ] Progress tracking and logging

### Step 4: Add Configuration & Metadata
- [ ] Configuration file
- [ ] Metadata generation
- [ ] README template

### Step 5: Testing & Validation
- [ ] Test on small subset
- [ ] Validate extracted data
- [ ] Performance benchmarking
- [ ] Error handling tests

## Usage Examples

### Interactive Selection (Notebook)
```python
# In notebook
from scripts.extract_favela_data import explore_favelas

# Launch interactive interface
selected = explore_favelas()
# Returns list of selected favela names
```

### Command Line - Single Favela
```bash
python scripts/extract_favela_data.py \
    --favela "Vidigal" \
    --output-dir data/vidigal_tls/raw
```

### Command Line - Batch Extraction
```bash
python scripts/extract_favela_data.py \
    --favela-list favelas_selected.json \
    --output-base data \
    --parallel \
    --workers 4
```

### Python API
```python
from scripts.extract_favela_data import extract_favela_data

result = extract_favela_data(
    favela_name="Vidigal",
    buildings_path="data/RJ/buildings_RJ_2019.shp",
    streets_path="data/RJ/Logradouros.shp",
    favelas_path="data/RJ/Favelas_Limit_2019.shp",
    output_dir="data/vidigal_tls/raw"
)
```

## File Naming Conventions

### Sanitization Rules
- Convert to lowercase
- Replace spaces with underscores
- Remove special characters
- Handle accented characters (normalize to ASCII)

### Examples
- "Vila da Conquista" → `vila_da_conquista`
- "São João" → `sao_joao`
- "Caminho Novo da Represa" → `caminho_novo_da_represa`

## Error Handling

### Common Issues & Solutions
1. **No buildings found**: Warn user, check boundary geometry
2. **No streets found**: Increase buffer, check street network coverage
3. **CRS mismatch**: Auto-detect and reproject
4. **Invalid geometry**: Fix with `buffer(0)` or `make_valid()`
5. **Memory errors**: Use chunked processing, reduce parallel workers

## Future Enhancements
- [ ] Support for additional data layers (DTM, STL meshes)
- [ ] Automatic STL mesh generation from building footprints
- [ ] Integration with existing analysis pipeline
- [ ] Web-based selection interface
- [ ] Data versioning and updates
